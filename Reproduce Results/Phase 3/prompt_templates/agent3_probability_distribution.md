You are given the set of marked states for an n-qubit Grover circuit. Grover's algorithm, after the correct number of iterations, concentrates almost all of the probability mass on the marked states, leaving a small residual amplitude on each of the remaining non-marked states. Your task is to classically simulate the final output probability distribution. The output should be a dictionary where each key is an n-bit string and each value is the corresponding probability rounded to four decimal places. Sort the entries in descending order by probability. If 2^n > 30, return only the top 30 most probable states — the t marked states first (each at p_marked), then, in ascending bitstring order, the smallest non-marked states (each at p_unmarked) until 30 entries are emitted.

Your response MUST end immediately after the line `=== END ===` that follows the closing `}` of the simulation-results dictionary. Do not produce any text after `=== END ===`. Do not repeat or re-emit the examples shown below.

=== Required reasoning structure ===

=== Probability Reasoning ===

  1. Marked-state count: t = <count>
  2. Qubit count: n = <n>, so state-space size N = 2^n = <2^n>
  3. Each marked state receives equal amplitude under Grover's algorithm. So, per marked state probability: p_marked = <p_marked>
  4. Remaining (N − t) = <N>−<t> = <N - t> non-marked states share the residual probability. Per non-marked state probability: p_unmarked = <p_unmarked> (very small positive number)
  5. Total probability is approximately 1: t · p_marked + (N − t) · p_unmarked = <t> · <p_marked> + (<N − t>) · <p_unmarked> = <sum>
  6. Verbatim marked-state strings (each length n = <n>):
     - <m_1>
     - <m_2>
     - ...
  7. Output schema: { '<n-bit state string>': <4-decimal prob>, ... } sorted in descending order by probability. I will emit min(N, 30) = min(<N>,30) = <result> entries total where:
 - first entries are <t> marked-state strings at <p_marked> probability
 - remaining entries are (min(N,30) - t) = (min(<N>,30) - <t>) = <result> un-marked-state strings at <p_unmarked> probability in ascending bitstring order

=== Simulation Results of the Grover's Algorithm ===
{
  '<state>': <prob>,
  ...
}
=== END ===

=== Examples ===

--- Example 1 (n=2, t=1, N=4 — no truncation) ---

Input:
=== Final Marked States ===
01

Output:
=== Probability Reasoning ===

  1. Marked-state count: t = 1
  2. Qubit count: n = 2, so state-space size N = 2^n = 4
  3. Each marked state receives equal amplitude under Grover's algorithm. So, per marked state probability: p_marked = 1.0000
  4. Remaining (N − t) = 4−1 = 3 non-marked states share the residual probability. Per non-marked state probability: p_unmarked = 0.0000 (very small positive number)
  5. Total probability is approximately 1: t · p_marked + (N − t) · p_unmarked = 1 · 1.0000 + 3 · 0.0000 = 1.0000
  6. Verbatim marked-state strings (each length n = 2):
     - 01
  7. Output schema: { '<2-bit state string>': <4-decimal prob>, ... } sorted in descending order by probability. I will emit min(N, 30) = min(4,30) = 4 entries total where:
 - first entries are 1 marked-state strings at 1.0000 probability
 - remaining entries are (min(N,30) - t) = (min(4,30) - 1) = 3 un-marked-state strings at 0.0000 probability in ascending bitstring order

=== Simulation Results of the Grover's Algorithm ===
{
  '01': 1.0000,
  '00': 0.0000,
  '10': 0.0000,
  '11': 0.0000
}
=== END ===

--- Example 2 (n=7, t=3, N=128 — top-30 truncation, decimal p_unmarked) ---

Input:
=== Final Marked States ===
0000001
0000101
1100011

Output:
=== Probability Reasoning ===

  1. Marked-state count: t = 3
  2. Qubit count: n = 7, so state-space size N = 2^n = 128
  3. Each marked state receives equal amplitude under Grover's algorithm. So, per marked state probability: p_marked = 0.3286
  4. Remaining (N − t) = 128−3 = 125 non-marked states share the residual probability. Per non-marked state probability: p_unmarked = 0.0001 (very small positive number)
  5. Total probability is approximately 1: t · p_marked + (N − t) · p_unmarked = 3 · 0.3286 + (125) · 0.0001 = 0.9983
  6. Verbatim marked-state strings (each length n = 7):
     - 0000001
     - 0000101
     - 1100011
  7. Output schema: { '<7-bit state string>': <4-decimal prob>, ... } sorted in descending order by probability. I will emit min(N, 30) = min(128,30) = 30 entries total where:
 - first entries are 3 marked-state strings at 0.3286 probability
 - remaining entries are (min(N,30) - t) = (min(128,30) - 3) = 27 un-marked-state strings at 0.0001 probability in ascending bitstring order

=== Simulation Results of the Grover's Algorithm ===
{
  '0000001': 0.3286,
  '0000101': 0.3286,
  '1100011': 0.3286,
  '0000000': 0.0001,
  '0000010': 0.0001,
  '0000011': 0.0001,
  '0000100': 0.0001,
  '0000110': 0.0001,
  '0000111': 0.0001,
  '0001000': 0.0001,
  '0001001': 0.0001,
  '0001010': 0.0001,
  '0001011': 0.0001,
  '0001100': 0.0001,
  '0001101': 0.0001,
  '0001110': 0.0001,
  '0001111': 0.0001,
  '0010000': 0.0001,
  '0010001': 0.0001,
  '0010010': 0.0001,
  '0010011': 0.0001,
  '0010100': 0.0001,
  '0010101': 0.0001,
  '0010110': 0.0001,
  '0010111': 0.0001,
  '0011000': 0.0001,
  '0011001': 0.0001,
  '0011010': 0.0001,
  '0011011': 0.0001,
  '0011100': 0.0001
}
=== END ===

--- Example 3 (n=6, t=2, N=64 — top-30 truncation, scientific-notation p_unmarked) ---

Input:
=== Final Marked States ===
000000
000111

Output:
=== Probability Reasoning ===

  1. Marked-state count: t = 2
  2. Qubit count: n = 6, so state-space size N = 2^n = 64
  3. Each marked state receives equal amplitude under Grover's algorithm. So, per marked state probability: p_marked = 0.4996
  4. Remaining (N − t) = 64−2 = 62 non-marked states share the residual probability. Per non-marked state probability: p_unmarked = 1.4452e-05 (very small positive number)
  5. Total probability is approximately 1: t · p_marked + (N − t) · p_unmarked = 2 · 0.4996 + (62) · 1.4452e-05 = 1.0001
  6. Verbatim marked-state strings (each length n = 6):
     - 000000
     - 000111
  7. Output schema: { '<6-bit state string>': <4-decimal prob>, ... } sorted in descending order by probability. I will emit min(N, 30) = min(64,30) = 30 entries total where:
 - first entries are 2 marked-state strings at 0.4996 probability
 - remaining entries are (min(N,30) - t) = (min(64,30) - 2) = 28 un-marked-state strings at 1.4452e-05 probability in ascending bitstring order

=== Simulation Results of the Grover's Algorithm ===
{
  '000000': 0.4996,
  '000111': 0.4996,
  '000001': 1.4452e-05,
  '000010': 1.4452e-05,
  '000011': 1.4452e-05,
  '000100': 1.4452e-05,
  '000101': 1.4452e-05,
  '000110': 1.4452e-05,
  '001000': 1.4452e-05,
  '001001': 1.4452e-05,
  '001010': 1.4452e-05,
  '001011': 1.4452e-05,
  '001100': 1.4452e-05,
  '001101': 1.4452e-05,
  '001110': 1.4452e-05,
  '001111': 1.4452e-05,
  '010000': 1.4452e-05,
  '010001': 1.4452e-05,
  '010010': 1.4452e-05,
  '010011': 1.4452e-05,
  '010100': 1.4452e-05,
  '010101': 1.4452e-05,
  '010110': 1.4452e-05,
  '010111': 1.4452e-05,
  '011000': 1.4452e-05,
  '011001': 1.4452e-05,
  '011010': 1.4452e-05,
  '011011': 1.4452e-05,
  '011100': 1.4452e-05,
  '011101': 1.4452e-05
}
=== END ===

The three examples above are for your reference only. Do not echo or re-emit them in your response. Produce only the analysis for the following input. End with `=== END ===` and stop.

=== Now perform the same task on the following input ===
Input:
{USER_INPUT}

Output:
