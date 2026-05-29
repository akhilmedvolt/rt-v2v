// AudioWorklet processor that converts the live mic Float32 stream into Int16
// PCM and posts each callback's buffer back to the main thread for batching +
// transmission over the WebSocket.
//
// At a 16 kHz AudioContext, `process` is invoked every 128 samples (~8 ms),
// which is a comfortable cadence for VAD on the server.

class PCMWorklet extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const channel = input[0];
    if (!channel || channel.length === 0) return true;

    const int16 = new Int16Array(channel.length);
    for (let i = 0; i < channel.length; i++) {
      let s = channel[i];
      if (s > 1) s = 1;
      else if (s < -1) s = -1;
      int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    this.port.postMessage(int16.buffer, [int16.buffer]);
    return true;
  }
}

registerProcessor('pcm-worklet', PCMWorklet);
