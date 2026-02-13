/**
 * wav_encoder.js - PCM Float32 → WAV Blob encoder
 * 
 * Encodes raw PCM audio data into WAV format for HTTP upload to HPC server.
 * Whisper requires 16kHz mono audio.
 */

class WavEncoder {
  /**
   * Encode Float32Array PCM samples into a WAV Blob.
   * @param {Float32Array} samples - Raw PCM samples
   * @param {number} sampleRate - Sample rate (default 16000)
   * @returns {Blob} WAV file as Blob
   */
  static encode(samples, sampleRate = 16000) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    // RIFF header
    WavEncoder._writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    WavEncoder._writeString(view, 8, 'WAVE');

    // fmt chunk
    WavEncoder._writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);      // chunk size
    view.setUint16(20, 1, true);       // PCM format
    view.setUint16(22, 1, true);       // mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true); // byte rate
    view.setUint16(32, 2, true);       // block align
    view.setUint16(34, 16, true);      // bits per sample

    // data chunk
    WavEncoder._writeString(view, 36, 'data');
    view.setUint32(40, samples.length * 2, true);

    // PCM samples (float32 → int16)
    for (let i = 0; i < samples.length; i++) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }

    return new Blob([buffer], { type: 'audio/wav' });
  }

  /**
   * Merge multiple Float32Array chunks into one.
   * @param {Float32Array[]} chunks
   * @returns {Float32Array}
   */
  static mergeChunks(chunks) {
    const totalLength = chunks.reduce((sum, c) => sum + c.length, 0);
    const result = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of chunks) {
      result.set(chunk, offset);
      offset += chunk.length;
    }
    return result;
  }

  static _writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }
}
