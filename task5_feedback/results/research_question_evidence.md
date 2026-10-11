# Task 5 research-question evidence

## RQ1 - Extra flexibility of AI feedback
The exact verifier's reasoning sensitivity is 0.0%, while the AI judge's is 0.0%. The AI filler wrong-preference rate is 25.0%. AI feedback is useful when it correctly separates reasoning quality inside verifier ties, but filler errors show that not every distinction is useful.

## RQ2 - Style, verbosity, and persuasive framing
On clean-versus-persuasive-filler pairs, the judge preferred the clean response 0.0% of the time and incorrectly preferred filler 25.0% of the time. Use the saved examples to determine whether the remaining decisions reflect verbosity, confidence, or another surface cue.

## RQ3 - Reward-source vulnerabilities
The AI judge's largest wrong-preference rate occurs for `persuasive_filler` at 25.0%. The verifier has S_reason=0.0% and S_outcome=100.0%: this contrast quantifies its outcome sensitivity and reasoning blindness.

## RQ4 - Out-of-domain transfer
SVAMP exact-accuracy ranking: rlvr > sft > rlaif. sft exact drop=-15.7%; rlvr exact drop=-16.0%; rlaif exact drop=-17.0%. Compare these changes with the saved AI-pairwise and response-length changes before making a transfer claim.
