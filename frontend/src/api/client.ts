import type { SimulationResponse, LiveCarbonResponse } from "@/types";

const BASE = (import.meta.env.VITE_API_URL as string | undefined) ?? "";

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`API ${res.status}`);
  return res.json() as Promise<T>;
}

// ── Request types ─────────────────────────────────────────────────────────

export interface RunSimulationRequest {
  strategies: string[];
  eval_days: number;
  checkpoint_name: string;
  seed: number;
  use_live: boolean;
}

// ── Typed API functions ───────────────────────────────────────────────────

export async function runSimulation(req: RunSimulationRequest): Promise<SimulationResponse> {
  return post<SimulationResponse>("/api/simulation/run", req);
}

export async function fetchLiveCarbon(): Promise<LiveCarbonResponse> {
  return get<LiveCarbonResponse>("/api/carbon/live");
}

