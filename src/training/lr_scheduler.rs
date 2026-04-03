
/// Simple learning rate scheduler
#[derive(Debug, Clone)]
pub struct LRScheduler {
    lr_max: f64,
    lr_min: f64,
    warmup_steps: usize,
    total_steps: usize,
    current_step: usize,
}

impl LRScheduler {
    pub fn new(lr_max: f64, lr_min: f64, warmup_steps: usize, total_steps: usize) -> Self {
        Self {
            lr_max,
            lr_min,
            warmup_steps,
            total_steps,
            current_step: 0,
        }
    }

    pub fn get_lr(&self) -> f64 {
        self.get_lr_at_step(self.current_step)
    }

    pub fn get_lr_at_step(&self, step: usize) -> f64 {
        if step < self.warmup_steps {
            let progress = step as f64 / self.warmup_steps as f64;
            self.lr_max * progress
        } else {
            let steps_after = step - self.warmup_steps;
            let total_after = self.total_steps - self.warmup_steps;
            let progress = steps_after as f64 / total_after as f64;
            let cosine = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
            self.lr_min + (self.lr_max - self.lr_min) * cosine
        }
    }

    pub fn step(&mut self) {
        self.current_step += 1;
    }

    pub fn set_step(&mut self, step: usize) {
        self.current_step = step;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_warmup() {
        let scheduler = LRScheduler::new(1.0, 0.1, 100, 1000);
        assert_eq!(scheduler.get_lr_at_step(0), 0.0);
        assert_eq!(scheduler.get_lr_at_step(50), 0.5);
        assert_eq!(scheduler.get_lr_at_step(100), 1.0);
    }
}

