from pyb import Pin, micros, elapsed_micros, delay

class IRRemote:
    def __init__(self, pin_name):
        
        self.pin = Pin(pin_name, Pin.IN)

    def _measure_pulse(self, target_level, timeout=20000):
        """
        测量引脚维持在 target_level 电平的时间 (us)
        :param target_level: 目标电平 (0 或 1)
        :param timeout: 超时时间 (us)
        :return: 持续时间 (us)，如果超时则返回 0
        """
        start = micros()
        # 1. 等待引脚变为目标电平 (同步信号)
        while self.pin.value() != target_level:
            if elapsed_micros(start) > timeout:
                return 0
        
        # 2. 计时开始
        pulse_start = micros()
        # 3. 等待引脚离开目标电平
        while self.pin.value() == target_level:
            if elapsed_micros(pulse_start) > timeout:
                return 0
        
        return elapsed_micros(pulse_start)

    def receive(self):
        """
        尝试接收红外信号
        :return: 解码后的 32位整数 (例如 0xFFA25D)，如果没有信号或解码失败返回 None
        """
        # --- 0. 快速检查 (非阻塞) ---
        # 红外接收头空闲时是高电平(1)。
        # 如果当前是高电平，说明没有信号来，直接返回，不要卡住主循环。
        if self.pin.value() == 1:
            return None

        # --- 1. 引导码检测 ---
        # NEC 协议开头：9ms 低电平
        pulse = self._measure_pulse(0)
        if not (8000 < pulse < 10000):
            return None

        # NEC 协议开头：4.5ms 高电平
        pulse = self._measure_pulse(1)
        if not (4000 < pulse < 5000):
            return None

        # --- 2. 数据读取 (32位) ---
        data = 0
        for i in range(32):
            # 每一位数据都以 560us 的低电平开始
            if self._measure_pulse(0, timeout=1000) == 0:
                return None

            # 接下来是高电平，时长决定是 0 还是 1
            # 逻辑 0: ~560us 高电平
            # 逻辑 1: ~1690us 高电平
            high_t = self._measure_pulse(1, timeout=3000)
            
            if high_t == 0: # 超时错
                return None

            # 阈值判定：大于 1000us 视为 1 (逻辑1通常是1600+，逻辑0是500+)
            if high_t > 1000:
                data |= (1 << i)
        
        return data
