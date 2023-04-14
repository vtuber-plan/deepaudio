
import tempfile
from . import logging
import os
from typing import Dict, Optional, Union
import warnings

from huggingface_hub import (
    CommitOperationAdd,
    create_commit,
    create_repo,
    get_hf_file_metadata,
    hf_hub_download,
    hf_hub_url,
    whoami,
)

logger = logging.get_logger(__name__)

HUGGINGFACE_CO_RESOLVE_ENDPOINT = "https://huggingface.co"

class PushToHubMixin:
    """
    A Mixin containing the functionality to push a model or tokenizer to the hub.
    """

    def _create_repo(
        self,
        repo_id: str,
        private: Optional[bool] = None,
        use_auth_token: Optional[Union[bool, str]] = None
    ) -> str:
        url = create_repo(repo_id=repo_id, token=use_auth_token, private=private, exist_ok=True)

        # If the namespace is not there, add it or `upload_file` will complain
        if "/" not in repo_id and url != f"{HUGGINGFACE_CO_RESOLVE_ENDPOINT}/{repo_id}":
            repo_id = get_full_repo_name(repo_id, token=use_auth_token)
        return repo_id

    def _get_files_timestamps(self, working_dir: Union[str, os.PathLike]):
        """
        Returns the list of files with their last modification timestamp.
        """
        return {f: os.path.getmtime(os.path.join(working_dir, f)) for f in os.listdir(working_dir)}

    def _upload_modified_files(
        self,
        working_dir: Union[str, os.PathLike],
        repo_id: str,
        files_timestamps: Dict[str, float],
        commit_message: Optional[str] = None,
        token: Optional[Union[bool, str]] = None,
        create_pr: bool = False,
    ):
        """
        Uploads all modified files in `working_dir` to `repo_id`, based on `files_timestamps`.
        """
        if commit_message is None:
            if "Model" in self.__class__.__name__:
                commit_message = "Upload model"
            elif "Config" in self.__class__.__name__:
                commit_message = "Upload config"
            elif "Tokenizer" in self.__class__.__name__:
                commit_message = "Upload tokenizer"
            elif "FeatureExtractor" in self.__class__.__name__:
                commit_message = "Upload feature extractor"
            elif "Processor" in self.__class__.__name__:
                commit_message = "Upload processor"
            else:
                commit_message = f"Upload {self.__class__.__name__}"
        modified_files = [
            f
            for f in os.listdir(working_dir)
            if f not in files_timestamps or os.path.getmtime(os.path.join(working_dir, f)) > files_timestamps[f]
        ]
        operations = []
        for file in modified_files:
            operations.append(CommitOperationAdd(path_or_fileobj=os.path.join(working_dir, file), path_in_repo=file))
        logger.info(f"Uploading the following files to {repo_id}: {','.join(modified_files)}")
        return create_commit(
            repo_id=repo_id, operations=operations, commit_message=commit_message, token=token, create_pr=create_pr
        )

    def push_to_hub(
        self,
        repo_id: str,
        use_temp_dir: Optional[bool] = None,
        commit_message: Optional[str] = None,
        private: Optional[bool] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
        max_shard_size: Optional[Union[int, str]] = "10GB",
        create_pr: bool = False,
        **deprecated_kwargs,
    ) -> str:
        # Deprecation warning will be sent after for repo_url and organization
        repo_url = deprecated_kwargs.pop("repo_url", None)
        organization = deprecated_kwargs.pop("organization", None)

        if os.path.isdir(repo_id):
            working_dir = repo_id
            repo_id = repo_id.split(os.path.sep)[-1]
        else:
            working_dir = repo_id.split("/")[-1]

        repo_id = self._create_repo(
            repo_id, private=private, use_auth_token=use_auth_token, repo_url=repo_url, organization=organization
        )

        if use_temp_dir is None:
            use_temp_dir = not os.path.isdir(working_dir)

        with tempfile.TemporaryDirectory() as work_dir:
            files_timestamps = self._get_files_timestamps(work_dir)

            # Save all files.
            self.save_pretrained(work_dir, max_shard_size=max_shard_size)

            return self._upload_modified_files(
                work_dir,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=use_auth_token,
                create_pr=create_pr,
            )

def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"
