from pyb import Pin, udelay, micros, elapsed_micros

class UltraSonic:
    def __init__(self, trig_id, echo_id, window=5, alpha=0.3, deadband=0.5):
        """
        :param trig_id: Trig 引脚名，如 'X1'
        :param echo_id: Echo 引脚名，如 'X3'
        :param window:  中值滤波窗口大小（建议奇数，如 3 或 5）
        :param alpha:   EMA平滑系数 (0~1)，越小越平滑，越大反应越快
        :param deadband: 死区阈值(cm)，变化小于此值不更新输出
        """
        # 硬件初始化
        self.trig = Pin(trig_id, Pin.OUT_PP)
        self.echo = Pin(echo_id, Pin.IN)
        self.trig.low()
        
        # 滤波参数
        self.window = window
        self.alpha = alpha
        self.deadband = deadband
        
        # 内部状态
        self.buf = []           # 历史数据缓冲
        self.ema_val = None     # EMA 当前值
        self.last_output = 0.0  # 上次输出的稳定值

    def _raw_measure(self):
        """底层原始测距 (私有方法)"""
        # 触发信号
        self.trig.high()
        udelay(10)
        self.trig.low()
        
        # 等待回响
        t0 = micros()
        while self.echo.value() == 0:
            if elapsed_micros(t0) > 30000: return None # 超时 30ms
        
        t1 = micros()
        while self.echo.value() == 1:
            if elapsed_micros(t1) > 30000: return None # 超时
            
        # 计算距离 (cm)
        dt = micros() - t1
        if dt < 0: return None 
        return dt / 58.0

    def distance(self):
        """
        获取平滑后的距离 (核心方法)
        :return: 稳定距离 (float)
        """
        # 1. 获取原始值
        raw = self._raw_measure()
        
        # 如果测距失败，直接返回上次的稳定值，保证数据连续性
        if raw is None:
            return self.last_output
            
        # 物理范围限制 (2cm ~ 400cm)，过滤离谱值
        raw = max(2, min(400, raw))

        # 2. 第一级：中值滤波 (去噪点)
        self.buf.append(raw)
        if len(self.buf) > self.window:
            self.buf.pop(0)
        
        # 只有缓冲满了才计算中值，否则暂时用 raw
        if len(self.buf) >= 3:
            # 排序后取中间那个数
            mid_val = sorted(self.buf)[len(self.buf)//2]
        else:
            mid_val = raw

        # 3. 第二级：EMA 滤波 (平滑趋势)
        if self.ema_val is None:
            self.ema_val = mid_val
        else:
            # 公式：新值 = alpha * 观测值 + (1-alpha) * 旧值
            self.ema_val = (self.alpha * mid_val) + ((1 - self.alpha) * self.ema_val)

        # 4. 第三级：死区判断 (静止防抖)
        # 只有变化量大于死区，才更新输出
        if abs(self.ema_val - self.last_output) > self.deadband:
            self.last_output = self.ema_val
                                                  
        return self.last_output
