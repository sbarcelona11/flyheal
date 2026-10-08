/** Clamp a temporal cursor when switching between episodes of different lengths. */
export function frameIndex(position, length) {
  const value = Number.isFinite(position) ? Math.floor(position) : 0;
  return Math.max(0, Math.min(value, Math.max(0, length - 1)));
}

/** Commit only the latest request, even when older responses arrive last. */
export function latestLoader(load, commit) {
  let sequence = 0;
  return async input => {
    const current = ++sequence;
    try {
      const data = await load(input);
      if (current === sequence) return await commit(data, input);
    } catch (error) {
      if (current === sequence) throw error;
    }
  };
}
