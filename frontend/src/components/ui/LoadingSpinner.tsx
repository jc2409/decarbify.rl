export default function LoadingSpinner() {
  return (
    <div className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-surface/80 backdrop-blur-sm">
      <div
        className="w-12 h-12 rounded-full border-4 border-surface-raised border-t-green-accent animate-spin mb-4"
      />
      <p className="text-white/70 text-sm font-medium">Running simulation…</p>
    </div>
  );
}
