//! Opt-in, lifetime-bound Wayland input-idle observer.
//!
//! Uses ext-idle-notify-v1 version 2 to ignore idle inhibitors. The compositor
//! reports only idle/resumed transitions; this client never captures keys,
//! mouse positions, application titles, or raw input events.

use std::{
    os::fd::AsRawFd,
    sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use anyhow::{Result, Context, bail};
use wayland_client::{
    Connection, Dispatch, QueueHandle, globals::{registry_queue_init, GlobalListContents},
    protocol::{wl_registry, wl_seat},
};
use wayland_protocols::ext::idle_notify::v1::client::{
    ext_idle_notification_v1::{self, ExtIdleNotificationV1},
    ext_idle_notifier_v1::ExtIdleNotifierV1,
};

const IDLE_THRESHOLD: Duration = Duration::from_millis(1_000);

#[derive(Default)]
struct Reading {
    connected_at: Option<Instant>,
    idle_since: Option<Instant>,
}

struct WaylandState {
    reading: Arc<Mutex<Reading>>,
}

impl Dispatch<wl_registry::WlRegistry, GlobalListContents> for WaylandState {
    fn event(
        _state: &mut Self, _proxy: &wl_registry::WlRegistry,
        _event: wl_registry::Event, _data: &GlobalListContents,
        _conn: &Connection, _qh: &QueueHandle<Self>,
    ) {}
}

impl Dispatch<wl_seat::WlSeat, ()> for WaylandState {
    fn event(
        _state: &mut Self, _proxy: &wl_seat::WlSeat,
        _event: wl_seat::Event, _data: &(),
        _conn: &Connection, _qh: &QueueHandle<Self>,
    ) {}
}

impl Dispatch<ExtIdleNotifierV1, ()> for WaylandState {
    fn event(
        _state: &mut Self, _proxy: &ExtIdleNotifierV1,
        _event: wayland_protocols::ext::idle_notify::v1::client::ext_idle_notifier_v1::Event,
        _data: &(), _conn: &Connection, _qh: &QueueHandle<Self>,
    ) {}
}

impl Dispatch<ExtIdleNotificationV1, ()> for WaylandState {
    fn event(
        state: &mut Self, _proxy: &ExtIdleNotificationV1,
        event: ext_idle_notification_v1::Event, _data: &(),
        _conn: &Connection, _qh: &QueueHandle<Self>,
    ) {
        let mut reading = state.reading.lock().unwrap_or_else(|e| e.into_inner());
        match event {
            ext_idle_notification_v1::Event::Idled => {
                reading.idle_since = Some(Instant::now());
            }
            ext_idle_notification_v1::Event::Resumed => {
                reading.idle_since = None;
            }
            _ => {}
        }
    }
}

fn watch(reading: Arc<Mutex<Reading>>, stop: Arc<AtomicBool>) -> Result<()> {
    let conn = Connection::connect_to_env().context("connecting to Wayland compositor")?;
    let (globals, mut queue) = registry_queue_init::<WaylandState>(&conn)?;
    let qh = queue.handle();
    let seat: wl_seat::WlSeat = globals.bind(&qh, 1..=9, ())?;
    // Version 2 provides get_input_idle_notification; this is deliberately
    // stricter than inhibitor-aware idle notifications from version 1.
    let notifier: ExtIdleNotifierV1 = globals.bind(&qh, 2..=2, ())
        .context("ext-idle-notify-v1 version 2 is unavailable")?;
    let _notification = notifier.get_input_idle_notification(
        IDLE_THRESHOLD.as_millis() as u32, &seat, &qh, (),
    );
    let mut state = WaylandState { reading: Arc::clone(&reading) };
    queue.roundtrip(&mut state)?;
    {
        let mut observed = reading.lock().unwrap_or_else(|e| e.into_inner());
        observed.connected_at = Some(Instant::now());
    }
    tracing::info!(
        target: "attent_activity::idle",
        provider = "wayland_ext_idle_notify_v2",
        threshold_ms = IDLE_THRESHOLD.as_millis() as u64,
        "input idle observer connected (no raw input captured)"
    );

    while !stop.load(Ordering::Acquire) {
        queue.flush()?;
        queue.dispatch_pending(&mut state)?;
        let Some(guard) = queue.prepare_read() else {
            continue;
        };
        let fd = guard.connection_fd().as_raw_fd();
        let mut event = libc::pollfd { fd, events: libc::POLLIN, revents: 0 };
        // Poll with a finite timeout so dropping the observer always stops
        // the OS input signal collection promptly on Stop.
        let rc = unsafe { libc::poll(&mut event, 1, 150) };
        if rc > 0 && (event.revents & libc::POLLIN) != 0 {
            let _ = guard.read()?;
        } else if rc < 0 {
            let err = std::io::Error::last_os_error();
            if err.kind() != std::io::ErrorKind::Interrupted {
                bail!("polling Wayland idle events failed: {err}");
            }
        }
        // Otherwise the read guard is dropped without reading the socket.
    }
    Ok(())
}

pub struct InputIdleMonitor {
    reading: Arc<Mutex<Reading>>,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl InputIdleMonitor {
    pub fn start() -> Self {
        let reading = Arc::new(Mutex::new(Reading::default()));
        let stop = Arc::new(AtomicBool::new(false));
        let worker_reading = Arc::clone(&reading);
        let worker_stop = Arc::clone(&stop);
        let handle = thread::Builder::new()
            .name("attent-input-idle".into())
            .spawn(move || {
                if let Err(err) = watch(Arc::clone(&worker_reading), worker_stop) {
                    tracing::info!(
                        target: "attent_activity::idle", error = %err,
                        "input idle observer unavailable; state will be Unknown"
                    );
                }
                let mut r = worker_reading.lock().unwrap_or_else(|e| e.into_inner());
                *r = Reading::default();
            })
            .ok();
        Self { reading, stop, handle }
    }

    pub fn idle_for(&self) -> Option<Duration> {
        let r = self.reading.lock().unwrap_or_else(|e| e.into_inner());
        let connected_at = r.connected_at?;
        // A new notification starts non-idle even before seeing user input.
        // Do not promote that initialization window to activity evidence.
        if let Some(since) = r.idle_since {
            return Some(IDLE_THRESHOLD + since.elapsed());
        }
        if connected_at.elapsed() >= IDLE_THRESHOLD + Duration::from_millis(250) {
            Some(Duration::ZERO)
        } else {
            None
        }
    }

    pub fn source(&self) -> Option<&'static str> {
        let r = self.reading.lock().unwrap_or_else(|e| e.into_inner());
        r.connected_at.map(|_| "wayland_ext_idle_notify_v2")
    }
}

impl Drop for InputIdleMonitor {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(handle) = self.handle.take() {
            if handle.join().is_err() {
                tracing::warn!(target: "attent_activity::idle", "idle observer thread panicked");
            }
        }
    }
}
