def build_rule_policy(candidates):
    """
    Static rule-based policy.

    For every stage, permanently choose the candidate
    with the lowest long-run P95 latency.
    """

    policy = {}

    for stage in candidates["stage"].unique():

        stage_candidates = candidates[
            candidates["stage"] == stage
        ]

        best_candidate = stage_candidates.loc[
            stage_candidates["level_p95"].idxmin()
        ]

        policy[stage] = int(best_candidate["cand_rank"])

    return policy


def choose_candidate(policy, stage):
    """
    Return the candidate selected by the static rule.
    """

    return policy[stage]