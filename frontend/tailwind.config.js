/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        cyber: {
          bg:      "#0a0c10",
          surface: "#111318",
          raised:  "#171b22",
          border:  "rgba(0,255,160,0.14)",
          green:   "#00FFA0",
          teal:    "#00D4FF",
          orange:  "#FF8C42",
          red:     "#FF4757",
          dim:     "#3a3f4b",
          muted:   "#6b7280",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["'Roboto Mono'", "monospace"],
      },
      boxShadow: {
        "glow-green": "0 0 20px rgba(0,255,160,0.25), 0 0 60px rgba(0,255,160,0.10)",
        "glow-teal":  "0 0 20px rgba(0,212,255,0.25), 0 0 60px rgba(0,212,255,0.10)",
        "glow-sm":    "0 0 8px rgba(0,255,160,0.30)",
      },
    },
  },
  plugins: [],
};
