export type Role = "user" | "assistant";

export interface Message {
  id: string;
  role: Role;
  /** For user messages, the QASM source. For assistant messages, the accumulating
   * stream content. */
  content: string;
  /** Set on user messages: the uploaded filename. */
  filename?: string;
  /** Set on assistant messages while the stream is still in flight. */
  streaming?: boolean;
  /** Set on assistant messages when the stream errored out. */
  error?: string;
}

export interface ApiSettings {
  baseUrl: string;
  modelName: string;
}
