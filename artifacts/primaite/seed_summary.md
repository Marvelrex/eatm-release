# Benchmark summary across seed rounds

Rounds: seed2025, seed42, seed818. Delta = arm mean minus same-round baseline mean, all over the first 5 from-scratch episodes (penalty-only reward, 0 is best, higher delta = better). Raw rewards are not comparable across rounds.

## multi

### EATM (Action-Graph tree + LLM chooser)

| round | sleep (worst case) | baseline mean | qwen3.7-flash | deepseek-v4-flash-0731 | glm-5.2 | gpt-5.6-terra | claude-opus-5 |
|---|---|---|---|---|---|---|---|
| seed2025 | -13,208 | -11,434 | +4,681 | +5,462 | +4,925 | +6,237 | +7,018 |
| seed42 | -13,197 | -10,438 | +3,613 | +4,356 | +4,029 | +4,153 | +6,069 |
| seed818 | -13,040 | -10,515 | +4,653 | +3,721 | +5,197 | +4,749 | +6,383 |

| model | mean delta | min..max | seeds |
|---|---|---|---|
| claude-opus-5 | +6,490 | +6,069..+7,018 | 3 |
| gpt-5.6-terra | +5,046 | +4,153..+6,237 | 3 |
| glm-5.2 | +4,717 | +4,029..+5,197 | 3 |
| deepseek-v4-flash-0731 | +4,513 | +3,721..+5,462 | 3 |
| qwen3.7-flash | +4,316 | +3,613..+4,681 | 3 |

### ACD pure-LLM (native llms-are-acd interface)

| round | sleep (worst case) | baseline mean | qwen3.7-flash | deepseek-v4-flash-0731 | glm-5.2 | gpt-5.6-terra | claude-opus-5 |
|---|---|---|---|---|---|---|---|
| seed2025 | -13,208 | -11,434 | +3,216 | +3,502 | +4,185 | +4,467 | +4,551 |
| seed42 | -13,197 | -10,438 | +2,309 | +2,470 | +3,083 | +3,467 | +3,399 |
| seed818 | -13,040 | -10,515 | +2,485 | +2,599 | +3,364 | +3,688 | +3,499 |

| model | mean delta | min..max | seeds |
|---|---|---|---|
| gpt-5.6-terra | +3,874 | +3,467..+4,467 | 3 |
| claude-opus-5 | +3,816 | +3,399..+4,551 | 3 |
| glm-5.2 | +3,544 | +3,083..+4,185 | 3 |
| deepseek-v4-flash-0731 | +2,857 | +2,470..+3,502 | 3 |
| qwen3.7-flash | +2,670 | +2,309..+3,216 | 3 |

### EATM vs ACD (cross-seed mean delta, same baseline)

| model | EATM | ACD | EATM advantage |
|---|---|---|---|
| qwen3.7-flash | +4,316 | +2,670 | +1,646 |
| deepseek-v4-flash-0731 | +4,513 | +2,857 | +1,655 |
| glm-5.2 | +4,717 | +3,544 | +1,173 |
| gpt-5.6-terra | +5,046 | +3,874 | +1,172 |
| claude-opus-5 | +6,490 | +3,816 | +2,674 |

## single

### EATM (Action-Graph tree + LLM chooser)

| round | sleep (worst case) | baseline mean | qwen3.7-flash | deepseek-v4-flash-0731 | glm-5.2 | gpt-5.6-terra | claude-opus-5 |
|---|---|---|---|---|---|---|---|
| seed2025 | -10,286 | -8,264 | +899 | +1,331 | +2,257 | +1,969 | +2,243 |
| seed42 | -10,439 | -10,311 | +3,760 | +2,451 | +3,456 | +3,731 | +3,707 |
| seed818 | -10,540 | -10,505 | +3,507 | +2,451 | +3,714 | +4,057 | +4,480 |

| model | mean delta | min..max | seeds |
|---|---|---|---|
| claude-opus-5 | +3,476 | +2,243..+4,480 | 3 |
| gpt-5.6-terra | +3,252 | +1,969..+4,057 | 3 |
| glm-5.2 | +3,143 | +2,257..+3,714 | 3 |
| qwen3.7-flash | +2,722 | +899..+3,760 | 3 |
| deepseek-v4-flash-0731 | +2,078 | +1,331..+2,451 | 3 |

### ACD pure-LLM (native llms-are-acd interface)

| round | sleep (worst case) | baseline mean | qwen3.7-flash | deepseek-v4-flash-0731 | glm-5.2 | gpt-5.6-terra | claude-opus-5 |
|---|---|---|---|---|---|---|---|
| seed2025 | -10,286 | -8,264 | -1,371 | -1,444 | -1,140 | -1,184 | -1,228 |
| seed42 | -10,439 | -10,311 | +572 | +465 | +717 | +744 | +606 |
| seed818 | -10,540 | -10,505 | +608 | +623 | +821 | +748 | +759 |

| model | mean delta | min..max | seeds |
|---|---|---|---|
| glm-5.2 | +133 | -1,140..+821 | 3 |
| gpt-5.6-terra | +103 | -1,184..+748 | 3 |
| claude-opus-5 | +46 | -1,228..+759 | 3 |
| qwen3.7-flash | -64 | -1,371..+608 | 3 |
| deepseek-v4-flash-0731 | -119 | -1,444..+623 | 3 |

### EATM vs ACD (cross-seed mean delta, same baseline)

| model | EATM | ACD | EATM advantage |
|---|---|---|---|
| qwen3.7-flash | +2,722 | -64 | +2,786 |
| deepseek-v4-flash-0731 | +2,078 | -119 | +2,196 |
| glm-5.2 | +3,143 | +133 | +3,010 |
| gpt-5.6-terra | +3,252 | +103 | +3,149 |
| claude-opus-5 | +3,476 | +46 | +3,431 |
