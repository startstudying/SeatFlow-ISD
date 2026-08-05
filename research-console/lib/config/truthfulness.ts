/**
 * Truthfulness configuration for the SeatFlow Research Console.
 *
 * All UI components must respect these constraints. The console must never
 * show "formal model" badges or "deployed" status.
 */

export interface TruthfulnessConfig {
  /** Model type must always be "Research / Backtest" */
  readonly modelType: "Research / Backtest";
  /** No deployment-related UI elements allowed */
  readonly deploymentEnabled: false;
  /** No live trading features */
  readonly liveTradingEnabled: false;
  /** Predictions are never formal */
  readonly formalPredictionsEnabled: false;
  /** Only paper trading allowed */
  readonly paperTradingEnabled: true;
  /** Blind evaluation is enforced */
  readonly blindEvaluationEnforced: true;
}

export const TRUTHFULNESS: TruthfulnessConfig = {
  modelType: "Research / Backtest",
  deploymentEnabled: false,
  liveTradingEnabled: false,
  formalPredictionsEnabled: false,
  paperTradingEnabled: true,
  blindEvaluationEnforced: true,
} as const;

/**
 * Check if a UI element is allowed under truthfulness constraints.
 */
export function isUIElementAllowed(element: string): boolean {
  const forbidden = [
    "deploy-button",
    "production-status",
    "live-trade-button",
    "formal-model-badge",
    "deployment-status",
    "real-money-warning",
  ];
  return !forbidden.includes(element);
}

/**
 * Get the human-readable model type label.
 */
export function getModelTypeLabel(): string {
  return TRUTHFULNESS.modelType;
}

/**
 * Format a prediction for display, ensuring truthfulness.
 */
export function formatPredictionLabel(prediction: {
  isFormal: boolean;
  isBlind: boolean;
}): string {
  if (prediction.isBlind) {
    return "Blind Prediction (Research)";
  }
  return "Research Prediction";
}
