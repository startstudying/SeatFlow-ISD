/**
 * SSE (Server-Sent Events) client for real-time task updates.
 *
 * Listens to the Control Center SSE stream and provides typed events.
 */

export type TaskEventType =
  | "TASK_CREATED"
  | "TASK_QUEUED"
  | "TASK_STARTED"
  | "TASK_PROGRESS"
  | "TASK_COMPLETED"
  | "TASK_FAILED"
  | "TASK_CANCELLED"
  | "TASK_BLOCKED";

export interface TaskEvent {
  eventType: TaskEventType;
  taskId: string;
  timestamp: string;
  traceId: string;
  payload: Record<string, unknown>;
}

export type TaskEventHandler = (event: TaskEvent) => void;

const SSE_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8080/api/v1";

/**
 * Subscribe to SSE events for a specific task.
 *
 * Returns an unsubscribe function.
 */
export function subscribeToTask(
  taskId: string,
  handler: TaskEventHandler,
  onError?: (error: Error) => void,
): () => void {
  const url = `${SSE_BASE}/tasks/${taskId}/events`;
  const eventSource = new EventSource(url, { withCredentials: true });

  eventSource.onmessage = (raw) => {
    try {
      const event = JSON.parse(raw.data) as TaskEvent;
      handler(event);
    } catch (e) {
      console.warn("Failed to parse SSE event:", e);
    }
  };

  eventSource.onerror = () => {
    const error = new Error(`SSE connection error for task ${taskId}`);
    if (onError) {
      onError(error);
    }
    // Auto-reconnect is handled by the browser's EventSource
  };

  return () => {
    eventSource.close();
  };
}

/**
 * Subscribe to all task events (global stream).
 */
export function subscribeToAllTasks(
  handler: TaskEventHandler,
  onError?: (error: Error) => void,
): () => void {
  const url = `${SSE_BASE}/tasks/events`;
  const eventSource = new EventSource(url, { withCredentials: true });

  eventSource.onmessage = (raw) => {
    try {
      const event = JSON.parse(raw.data) as TaskEvent;
      handler(event);
    } catch (e) {
      console.warn("Failed to parse SSE event:", e);
    }
  };

  eventSource.onerror = () => {
    const error = new Error("SSE global connection error");
    if (onError) {
      onError(error);
    }
  };

  return () => {
    eventSource.close();
  };
}

/**
 * React hook for task event subscription.
 */
export function useTaskEvents(taskId: string): {
  events: TaskEvent[];
  connected: boolean;
} {
  // This is a placeholder; the real implementation uses React state
  // with useRef for the EventSource and useState for events.
  return { events: [], connected: false };
}
