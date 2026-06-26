You are given a quantum circuit written in OpenQASM 3.0 that implements Grover's algorithm. Your task is to locate the block beginning with `gate Oracle _gate_q_0, _gate_q_1, …` and copy its body verbatim — preserving every X gate, every mcmt gate and the original ordering. You must not add operations, remove operations, reorder operations, change qubit indices, or interpret what the Oracle does. Your reasoning should follow the structure shown below.

Your response MUST end immediately after the line `=== END ===` that follows the closing `}` of the extracted Oracle gate definition. Do not produce any text after `=== END ===`. Do not repeat or re-emit the examples shown below.

=== Required reasoning structure ===

=== Oracle Extraction Reasoning ===

1. Locating the 'gate Oracle' definition in the input.
2. Each MCMT gate in the Oracle body, together with its surrounding X gates, encodes one marked state.
3. X-gate blocks must be consistent. For each MCMT, the X-gate sequence immediately before it must be the same as the X-gate sequence immediately after it (these X gates uncompute each other).
4. Extracted Oracle gate definition must be copied verbatim from input, with no modifications, no truncation, no extra operations.

=== Extracted Oracle Gate Definition ===
gate Oracle _gate_q_0, _gate_q_1, ..., _gate_q_<n-1> {
  <body copied verbatim from input>
}
=== END ===

=== Examples ===

--- Example 1 (n=2, 1 marked state) ---

Input:
OPENQASM 3.0;
include "stdgates.inc";
gate mcmt _gate_q_0, _gate_q_1 {
  cz _gate_q_0, _gate_q_1;
}
gate Oracle _gate_q_0, _gate_q_1 {
  x _gate_q_1;
  mcmt _gate_q_0, _gate_q_1;
  x _gate_q_1;
}
gate Diffuser _gate_q_0, _gate_q_1 {
  h _gate_q_0;
  h _gate_q_1;
  x _gate_q_0;
  x _gate_q_1;
  h _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  h _gate_q_1;
  x _gate_q_0;
  x _gate_q_1;
  h _gate_q_0;
  h _gate_q_1;
}
bit[2] c;
qubit[2] q;
h q[0];
h q[1];
Oracle q[0], q[1];
Diffuser q[0], q[1];
c[0] = measure q[0];
c[1] = measure q[1];

Output:
=== Oracle Extraction Reasoning ===

1. Locating the 'gate Oracle' definition in the input.
2. Each MCMT gate in the Oracle body, together with its surrounding X gates, encodes one marked state.
3. X-gate blocks must be consistent. The single MCMT has `x _gate_q_1` before it and `x _gate_q_1` after it — matched.
4. Extracted Oracle gate definition must be copied verbatim from input, with no modifications, no truncation, no extra operations.

=== Extracted Oracle Gate Definition ===
gate Oracle _gate_q_0, _gate_q_1 {
  x _gate_q_1;
  mcmt _gate_q_0, _gate_q_1;
  x _gate_q_1;
}
=== END ===

--- Example 2 (n=7, 3 marked states) ---

Input:
{EXAMPLE_N7_K3_QASM}

Output:
=== Oracle Extraction Reasoning ===

1. Locating the 'gate Oracle' definition in the input.
2. Each MCMT gate in the Oracle body, together with its surrounding X gates, encodes one marked state.
3. X-gate blocks must be consistent. Block 1: X gates on q1,q2,q3,q4,q5,q6 before MCMT and same X gates after — matched. Block 2: X gates on q1,q3,q4,q5,q6 before MCMT and same X gates after — matched. Block 3: X gates on q2,q3,q4 before MCMT and same X gates after — matched.
4. Extracted Oracle gate definition must be copied verbatim from input, with no modifications, no truncation, no extra operations.

=== Extracted Oracle Gate Definition ===
gate Oracle _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4, _gate_q_5, _gate_q_6 {
  x _gate_q_1;
  x _gate_q_2;
  x _gate_q_3;
  x _gate_q_4;
  x _gate_q_5;
  x _gate_q_6;
  mcmt _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4, _gate_q_5, _gate_q_6;
  x _gate_q_1;
  x _gate_q_2;
  x _gate_q_3;
  x _gate_q_4;
  x _gate_q_5;
  x _gate_q_6;
  x _gate_q_1;
  x _gate_q_3;
  x _gate_q_4;
  x _gate_q_5;
  x _gate_q_6;
  mcmt _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4, _gate_q_5, _gate_q_6;
  x _gate_q_1;
  x _gate_q_3;
  x _gate_q_4;
  x _gate_q_5;
  x _gate_q_6;
  x _gate_q_2;
  x _gate_q_3;
  x _gate_q_4;
  mcmt _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4, _gate_q_5, _gate_q_6;
  x _gate_q_2;
  x _gate_q_3;
  x _gate_q_4;
}
=== END ===

The two examples above are for your reference only. Do not echo or re-emit them in your response. Produce only the analysis for the following input. End with `=== END ===` and stop.

=== Now perform the same task on the following input ===
Input:
{USER_INPUT}

Output:
