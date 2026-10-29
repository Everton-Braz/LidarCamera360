# Contributing

Thanks for helping. A few ground rules keep the project easy to build on.

## License of contributions

The original LidarCamera360 source is MIT licensed (see `LICENSE`). By opening a pull request you
agree that your contribution is licensed under the MIT license, and you confirm that you wrote it
or have the right to submit it under those terms. Add a `Signed-off-by:` line to your commits
(`git commit -s`) to certify the [Developer Certificate of Origin](https://developercertificate.org/).

Do not paste code from GPL projects (FAST-LIVO2, MOLA, Spirula, PyQt examples, ...) into the MIT
source. Changes to the vendored `FAST-LIVO2/` port stay under its GPL-2.0-only license.

## Workflow

- Branch from `dev` and open the pull request against `dev`.
- Run `python -m pytest tests` before submitting.
- Keep the pipeline rig-agnostic: scanner or camera specifics belong in a rig profile, not in code.
