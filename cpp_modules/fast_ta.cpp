#include <vector>
#include <numeric>
#include <cmath>
extern "C" {
    // Basit EMA hesaplama
    double ema(const double* data, int len, int period) {
        if (len < period) return 0.0;
        double multiplier = 2.0 / (period + 1);
        double ema = data[0];
        for (int i = 1; i < len; ++i) {
            ema = (data[i] - ema) * multiplier + ema;
        }
        return ema;
    }
    // Basit RSI hesaplama
    double rsi(const double* data, int len, int period) {
        if (len <= period) return 0.0;
        double gain = 0.0, loss = 0.0;
        for (int i = 1; i <= period; ++i) {
            double diff = data[i] - data[i-1];
            if (diff > 0) gain += diff;
            else loss -= diff;
        }
        gain /= period;
        loss /= period;
        for (int i = period+1; i < len; ++i) {
            double diff = data[i] - data[i-1];
            if (diff > 0) gain = (gain * (period-1) + diff) / period;
            else loss = (loss * (period-1) - diff) / period;
        }
        if (loss == 0) return 100.0;
        double rs = gain / loss;
        return 100.0 - (100.0 / (1.0 + rs));
    }
    // Basit MACD hesaplama (son değer)
    double macd(const double* data, int len, int fast, int slow, int signal) {
        if (len < slow) return 0.0;
        double ema_fast = ema(data, len, fast);
        double ema_slow = ema(data, len, slow);
        double macd_val = ema_fast - ema_slow;
        // Signal line için tekrar EMA
        double macd_arr[1000];
        for (int i = 0; i < len; ++i) {
            macd_arr[i] = ema(&data[i], len-i, fast) - ema(&data[i], len-i, slow);
        }
        double signal_val = ema(macd_arr, len, signal);
        return macd_val - signal_val;
    }
} 