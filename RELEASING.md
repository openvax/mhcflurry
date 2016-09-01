# Releasing MHCflurry

We currently don't have a PyPI release but that will change soon (and these instructions will be updated indicating how to push updates). For now the only "releasing" to deal with is changing the downloadable models and data.

## Changing the downloadable models or data

We publish our downloadable models and data as files associated with a GitHub release. Since we need to refer to the URLs for these files in [downloads.yml](mhcflurry/downloads.yml), which is checked into the repo, updating the models requires a few steps: we have to make a preliminary GitHub release containing the new models and data as GitHub attached files, update the code to point to these files, wait for travis to run, merge the PR, then modify the release's tag to now point to the new master. Here are these steps in more detail:

* Make a new release by going [here](https://github.com/hammerlab/mhcflurry/releases/new). The tag should be the version of MHCflurry you are releasing. Make sure you check "This is a pre-release." It actually doesn't matter what commit the tag is associated with as we will change it later, but you might as well make it point to HEAD of the branch you are working from. Attach your generated downloads to the release.

* Modify [downloads.yml](mhcflurry/downloads.yml) to point to the URLs of your files above. Commit and push your changes to your branch.

* When travis has suceeded and code review is complete, merge your PR to master.

* Now *change* the GitHub release's tag to point to the current latest master. Based on this stackoverflow [answer](http://stackoverflow.com/questions/24849362/change-connected-commit-on-release-github) you can run (from a checkout of the updated master branch):

```
git tag -f -a 0.0.1
git push -f --tags
```

