"""Exact token-length adapter for verl V1's external reward-manager interface.

Target verl: jialei777/verl-upstream@9924801779415f86c807b5716a3d4479fa60f811.
This single-turn manager preserves NaiveRewardManager's decoding and output
contract, but passes the original completion-token count (including EOS) to the
vendored boxed_math_reward.py scorer. It never re-tokenizes decoded responses.

Configure:
  reward.reward_manager.source=importlib
  reward.reward_manager.name=BoxedRewardManager
  reward.reward_manager.module.path=/absolute/path/boxed_reward_v1.py
  reward.custom_reward_function.path=/absolute/path/boxed_math_reward.py
  reward.custom_reward_function.name=compute_score

Propagate REWARD_MAX_RESP_LEN=2048, REWARD_OVERLONG_BUFFER=512,
REWARD_OVERLONG_PENALTY=1.0, REWARD_FORMAT_SCORE=0.1 and
REWARD_PENALTY_SOURCES=gsm8k_boxed_train into the Ray runtime environment.
The source-based scope leaves gsm8k_boxed_test evaluation unpenalized.
"""

from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager


class BoxedRewardManager(NaiveRewardManager):
    """Naive single-turn reward computation with an exact completion length."""

    async def run_single(self, data):
        # Match the upstream manager: only the last sequence is scored.
        item = data[-1:][0]
        response_ids = item.batch["responses"]
        width = response_ids.shape[-1]
        valid_length = int(item.batch["attention_mask"][-width:].sum().item())
        valid_ids = response_ids[:valid_length]

        # A fresh dict avoids races between completions sharing prompt metadata.
        extra_info = dict(item.non_tensor_batch.get("extra_info") or {})
        tool_fields = item.non_tensor_batch.get("tool_extra_fields")
        if tool_fields is not None:
            extra_info.update(tool_fields)
        extra_info["num_turns"] = item.non_tensor_batch.get("__num_turns__")
        extra_info["rollout_reward_scores"] = item.non_tensor_batch.get("reward_scores", {})
        # Set this last so stale metadata/tool fields cannot override the count.
        extra_info["response_len"] = valid_length

        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_ids, skip_special_tokens=True)
        )
        kwargs = {
            "data_source": item.non_tensor_batch["data_source"],
            "solution_str": response_str,
            "ground_truth": item.non_tensor_batch["reward_model"]["ground_truth"],
            "extra_info": extra_info,
        }
        if self.reward_router_address is not None:
            kwargs.update(
                reward_router_address=self.reward_router_address,
                reward_model_tokenizer=self.reward_model_tokenizer,
            )
        if self.is_async_reward_score:
            result = await self.compute_score(**kwargs)
        else:
            result = await self.loop.run_in_executor(None, lambda: self.compute_score(**kwargs))
        if isinstance(result, dict):
            return {"reward_score": result["score"], "reward_extra_info": dict(result)}
        return {"reward_score": result, "reward_extra_info": {"acc": result}}
