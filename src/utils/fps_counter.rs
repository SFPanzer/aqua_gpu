use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

pub(crate) struct FpsCounter {
    frame_times: VecDeque<Duration>,
    last_frame_time: Instant,
    max_frames: usize,
    cached_fps: f32,
    last_update: Instant,
    update_interval: Duration,
}

impl FpsCounter {
    pub fn new(max_frames: usize, update_interval_secs: f32) -> Self {
        FpsCounter {
            frame_times: VecDeque::with_capacity(max_frames),
            last_frame_time: Instant::now(),
            max_frames,
            cached_fps: 0.0,
            last_update: Instant::now(),
            update_interval: Duration::from_secs_f32(update_interval_secs),
        }
    }

    pub fn tick(&mut self) {
        let now = Instant::now();
        let frame_time = now - self.last_frame_time;
        self.last_frame_time = now;

        self.frame_times.push_back(frame_time);
        if self.frame_times.len() > self.max_frames {
            self.frame_times.pop_front();
        };
    }

    pub fn fps(&mut self) -> f32 {
        let now = Instant::now();
        if now - self.last_update >= self.update_interval {
            self.update_fps();
            self.last_update = now;
        }
        self.cached_fps
    }

    fn update_fps(&mut self) {
        if self.frame_times.is_empty() {
            self.cached_fps = 0.0;
            return;
        }

        let total: Duration = self.frame_times.iter().sum();
        let average = total / (self.frame_times.len() as u32);
        self.cached_fps = 1.0 / average.as_secs_f32()
    }
}
