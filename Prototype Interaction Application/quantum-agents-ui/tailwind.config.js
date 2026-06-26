/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        // Single source of truth via CSS variables — themes swap via the `dark`
        // class on <html>. RGB triplets so Tailwind's opacity modifiers
        // (text-primary/70 etc.) still work.
        bg: {
          base:    "rgb(var(--bg-base) / <alpha-value>)",
          panel:   "rgb(var(--bg-panel) / <alpha-value>)",
          elev:    "rgb(var(--bg-elev) / <alpha-value>)",
          subtle:  "rgb(var(--bg-subtle) / <alpha-value>)",
        },
        border: {
          subtle:  "rgb(var(--border-subtle) / <alpha-value>)",
          strong:  "rgb(var(--border-strong) / <alpha-value>)",
        },
        text: {
          primary:   "rgb(var(--text-primary) / <alpha-value>)",
          secondary: "rgb(var(--text-secondary) / <alpha-value>)",
          muted:     "rgb(var(--text-muted) / <alpha-value>)",
        },
        // Per-agent semantic accents — cyan/violet/amber form a complete colour story.
        agent: {
          1: "rgb(var(--agent-1) / <alpha-value>)",
          2: "rgb(var(--agent-2) / <alpha-value>)",
          3: "rgb(var(--agent-3) / <alpha-value>)",
        },
        // General accents reused for header/branding/highlights.
        accent: {
          cyan:    "rgb(var(--accent-cyan)    / <alpha-value>)",
          violet:  "rgb(var(--accent-violet)  / <alpha-value>)",
          amber:   "rgb(var(--accent-amber)   / <alpha-value>)",
          magenta: "rgb(var(--accent-magenta) / <alpha-value>)",
        },
      },
      fontFamily: {
        sans: ['"Inter"', 'system-ui', 'sans-serif'],
        display: ['"Space Grotesk"', '"Inter"', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', '"Fira Code"', 'ui-monospace', 'monospace'],
      },
      animation: {
        "spin-slow": "spin 9s linear infinite",
        "aurora":    "aurora 18s ease-in-out infinite",
        "fade-in":   "fadeIn 240ms ease-out",
      },
      keyframes: {
        aurora: {
          "0%, 100%": { transform: "translate(0%, 0%) scale(1)" },
          "33%":      { transform: "translate(4%, -2%) scale(1.05)" },
          "66%":      { transform: "translate(-3%, 3%) scale(0.97)" },
        },
        fadeIn: {
          from: { opacity: 0, transform: "translateY(4px)" },
          to:   { opacity: 1, transform: "translateY(0)" },
        },
      },
      boxShadow: {
        "card":        "0 1px 0 0 rgb(var(--border-subtle)), 0 8px 24px -12px rgb(0 0 0 / 0.18)",
        "card-strong": "0 1px 0 0 rgb(var(--border-strong)), 0 12px 32px -8px rgb(0 0 0 / 0.32)",
      },
    },
  },
  plugins: [],
};
