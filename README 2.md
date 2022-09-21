[![CircleCI](https://circleci.com/gh/nauto/nauto-ai.svg?style=svg&circle-token=4e9dea414cd7476a8f7d24743bd635ff46c21c2a	)](https://circleci.com/gh/nauto/nauto-ai)

# nauto-ai
monorepo for nauto ml


## Update Submodules
* submodule manifests: https://github.com/nauto/nauto-ai/blob/master/.gitmodules
To update the `nauto-ai` submodules, we will need to create a new commit and run the following
```
>> cd <local-repo>/nauto-ai
>> git branch dummy-update-submodules
>> git checkout dummy-update-submodules
>> git commit -am "update all submodules"
>> git push
```

* One reason to update the git-submodule: `protobuf` repo is very important to Nauto AI engineering and we want access to the latest protobuf. In order for the team to access to the latest protobuf is to update the submodule(s).

Reference: https://stackoverflow.com/questions/8191299/update-a-submodule-to-the-latest-commit
