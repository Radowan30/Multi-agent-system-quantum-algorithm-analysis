You are given the Oracle gate definition extracted from a quantum circuit written in OpenQASM 3.0 that implements Grover's algorithm. The Oracle body consists of one or more mcmt (multi-controlled multi-target) gates, each preceded and followed by a matched set of X gates that uncompute each other. Each mcmt gate together with its surrounding X gates encodes exactly one marked state. Your task is to read the Oracle body, segment it into blocks (one per mcmt), derive the marked state for each block, and list all marked states.

Your response MUST end immediately after the line `=== END ===` that follows the last marked-state line. Do not produce any text after `=== END ===`. Do not repeat or re-emit the examples shown below.

=== Required reasoning structure ===

The Oracle entity is extracted below:

<oracle body, echoed verbatim>

There are <number> MCMT gates, so there are <number> Blocks.

=== Block <n> ===
Operation sequence:
<nth oracle entity block>

State construction:
<state construction logic>
Final state: <state>

[repeat for each block]

=== Final Marked States ===
<marked states, one per line>
=== END ===

=== Examples ===

--- Example 1 (n=2, 1 block, marks 01) ---

Input:
=== Extracted Oracle Gate Definition ===
gate Oracle _gate_q_0, _gate_q_1 {
  x _gate_q_1;
  mcmt _gate_q_0, _gate_q_1;
  x _gate_q_1;
}

Output:
The Oracle entity is extracted below:

gate Oracle _gate_q_0, _gate_q_1 {
  x _gate_q_1;
  mcmt _gate_q_0, _gate_q_1;
  x _gate_q_1;
}

There are 1 MCMT gates, so there are 1 Blocks.

=== Block 1 ===
Operation sequence:
x _gate_q_1;
mcmt _gate_q_0, _gate_q_1;
x _gate_q_1;

State construction:
x _gate_q_0: Absent → 1, then → 1
x _gate_q_1: Present → 0, then → 01
Final state: 01

=== Final Marked States ===
01
=== END ===

--- Example 2 (n=7, 3 blocks, marks 0000001, 0000101, 1100011) ---

Input:
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

Output:
The Oracle entity is extracted below:

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

There are 3 MCMT gates, so there are 3 Blocks.

=== Block 1 ===
Operation sequence:
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

State construction:
x _gate_q_0: Absent → 1, then → 1
x _gate_q_1: Present → 0, then → 01
x _gate_q_2: Present → 0, then → 001
x _gate_q_3: Present → 0, then → 0001
x _gate_q_4: Present → 0, then → 00001
x _gate_q_5: Present → 0, then → 000001
x _gate_q_6: Present → 0, then → 0000001
Final state: 0000001

=== Block 2 ===
Operation sequence:
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

State construction:
x _gate_q_0: Absent → 1, then → 1
x _gate_q_1: Present → 0, then → 01
x _gate_q_2: Absent → 1, then → 101
x _gate_q_3: Present → 0, then → 0101
x _gate_q_4: Present → 0, then → 00101
x _gate_q_5: Present → 0, then → 000101
x _gate_q_6: Present → 0, then → 0000101
Final state: 0000101

=== Block 3 ===
Operation sequence:
x _gate_q_2;
x _gate_q_3;
x _gate_q_4;
mcmt _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4, _gate_q_5, _gate_q_6;
x _gate_q_2;
x _gate_q_3;
x _gate_q_4;

State construction:
x _gate_q_0: Absent → 1, then → 1
x _gate_q_1: Absent → 1, then → 11
x _gate_q_2: Present → 0, then → 011
x _gate_q_3: Present → 0, then → 0011
x _gate_q_4: Present → 0, then → 00011
x _gate_q_5: Absent → 1, then → 100011
x _gate_q_6: Absent → 1, then → 1100011
Final state: 1100011

=== Final Marked States ===
0000001
0000101
1100011
=== END ===

The two examples above are for your reference only. Do not echo or re-emit them in your response. Produce only the analysis for the following input. End with `=== END ===` and stop.

=== Now perform the same task on the following input ===
Input:
{USER_INPUT}

Output:
