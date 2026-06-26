import { Binary, Calculator, Crosshair, Workflow, Zap } from "lucide-react";

const PIPELINE: Array<{
  n: 1 | 2 | 3;
  title: string;
  desc: string;
  Icon: typeof Workflow;
}> = [
  {
    n: 1,
    title: "Oracle Extractor",
    desc: "Locates the gate Oracle definition and copies it verbatim from the circuit.",
    Icon: Crosshair,
  },
  {
    n: 2,
    title: "Marked-State Identifier",
    desc: "Segments the oracle into MCMT blocks and derives each marked state.",
    Icon: Binary,
  },
  {
    n: 3,
    title: "Probability Distribution",
    desc: "Computes the Grover-amplified output distribution over basis states.",
    Icon: Calculator,
  },
];

export function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center text-center pt-10 pb-14 px-6 animate-fade-in">
      {/* Pipeline glyph — visually represents the three-stage agent chain */}
      <div className="relative mb-7">
        <div className="absolute inset-0 blur-3xl bg-accent-violet/25 dark:bg-accent-violet/30 rounded-full scale-150" />
        <div className="absolute inset-0 blur-2xl bg-accent-cyan/15 dark:bg-accent-cyan/25 rounded-full scale-125" />
        <Workflow className="relative w-16 h-16 text-accent-violet" strokeWidth={1.3} />
      </div>

      <h2 className="text-2xl sm:text-3xl font-display font-semibold tracking-tight bg-gradient-to-r from-accent-cyan via-accent-violet to-accent-amber bg-clip-text text-transparent mb-3">
        Multi-agent symbolic analysis
      </h2>
      <p className="text-text-secondary text-sm max-w-xl mb-10 leading-relaxed">
        Upload a Grover search circuit in OpenQASM 3.0. The system decomposes
        the analysis into three sequential agents — oracle extraction,
        marked-state identification, and probability distribution — and
        streams each agent's reasoning as it runs.
      </p>

      {/* Pipeline cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 sm:gap-4 max-w-3xl w-full">
        {PIPELINE.map((agent, i) => {
          const Icon = agent.Icon;
          return (
            <div
              key={agent.n}
              data-agent={agent.n}
              className="card px-4 py-4 text-left transition hover:shadow-card-strong"
              style={{ animationDelay: `${i * 90}ms` }}
            >
              <div className="flex items-center gap-2 mb-2">
                <div className="w-7 h-7 rounded-lg flex items-center justify-center agent-bg agent-border border">
                  <Icon className="w-3.5 h-3.5 agent-text" strokeWidth={2.2} />
                </div>
                <span className="chip agent-bg agent-text agent-border border">
                  Agent {agent.n}
                </span>
              </div>
              <h3 className="font-display font-semibold text-text-primary text-sm tracking-tight mb-1">
                {agent.title}
              </h3>
              <p className="text-xs text-text-secondary leading-relaxed">
                {agent.desc}
              </p>
            </div>
          );
        })}
      </div>

      <div className="mt-9 flex items-center gap-2 text-[11px] text-text-muted uppercase tracking-[0.18em]">
        <Zap className="w-3 h-3 text-accent-cyan" />
        <span>Local inference · Token-streamed · OpenAI-compatible</span>
      </div>
    </div>
  );
}
