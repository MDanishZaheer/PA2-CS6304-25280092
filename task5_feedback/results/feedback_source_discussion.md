# Feedback-source discussion

RLVR provides deterministic designated-final coverage with no model inference, but it is decisive on only 50.0% of the controlled pairs because equal final answers receive identical reward.

RLAIF is decisive on 13.7% of controlled pairs. Its wrong-preference rate is 7.5%, and its persuasive-filler wrong-preference rate is 25.0%. These quantities measure judge noise and surface-style exploitability rather than exact outcome specification.

The saved unique AI comparisons record approximately 360.3 seconds of judge loading plus inference. RLVR uses a local parser instead. Binary RLVR reward and AI pairwise reward are not calibrated utility scales and are not compared as raw magnitudes.
