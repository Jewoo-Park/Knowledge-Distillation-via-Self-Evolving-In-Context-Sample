"""Self-evolving in-context samples (ICS) for a frozen video-language student.

Per benchmark, questions arrive in blocks of 50. The student (Qwen3.5-9B) answers each question with the ICS
retrieved from that benchmark's memory; the teacher (gpt-6.1-sol, same 32 frames) solves each question blind.
After a block, the critic turns the block's cases into success / corrected ICS and edits the memory
(add / refine / merge / prune). The student's weights never change.
"""
