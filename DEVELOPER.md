# DEVELOPER.md

## Versioning

This library follows [Semantic Versioning](http://semver.org/).

## Processes

### Conventional Commit messages

This repository uses tool [Release Please](https://github.com/googleapis/release-please) to create GitHub and PyPi releases. It does so by parsing your
git history, looking for [Conventional Commit messages](https://www.conventionalcommits.org/),
and creating release PRs.

Learn more by reading [How should I write my commits?](https://github.com/googleapis/release-please?tab=readme-ov-file#how-should-i-write-my-commits)

### Modifying existing classes
When modifying any existing classes, ensure the required libraries are installed in "editable mode" using the command `pip install -e .` from your project's root directory.

This allows your code to dynamically reflect any changes you make to the library, enabling efficient local testing of your new methods.

## Testing

### Run tests locally

1. Set environment variables for `INSTANCE_ID`, `CLUSTER_ID`, `DATABASE_ID`, `DB_USER`, `DB_PASSWORD`, `OMNI_HOST`, `OMNI_USER`, `OMNI_PASSWORD`, `IAM_ACCOUNT`.

1. Run pytest to automatically run all tests:

    ```bash
    pytest
    ```

Notes:

* Tests run against public and private IP addresses. Tests for private IP can not be run locally due to VPC restrictions. There is no current way to prevent these tests from running. These tests will time out.
* Tests use both IAM and built-in authentication.
  * Learn how to set up a built-in databases user at [Manage AlloyDB user roles](https://cloud.google.com/alloydb/docs/database-users/about).
  * Local tests will run against your `gcloud` credentials. Use `gcloud` to login with your personal account or a service account. This account will be used to run IAM tests. Learn how to set up access to the database at [Manage IAM authentication](https://cloud.google.com/alloydb/docs/manage-iam-authn). The "IAM_ACCOUNT" environment variable is also used to test authentication to override the local account. A personal account or a service account can be used for this test.
  * You may need to grant access to the public schema for your new database user: `GRANT ALL ON SCHEMA public TO myaccount@example.com;`

### AlloyDB Omni Testing
The `ScaNN` index is an AlloyDB Omni preview and is not available on Cloud AlloyDB. To test for the `ScaNN` index integration, an AlloyDB Omni instance is set up and run on a GCE VM instance. The Omni instance is listening on input traffic on the private IP address of the VM. Integration tests related to the `ScaNN` index are only run against this Omni instance, while all other index tests are run against the Cloud AlloyDB testing instance.

For more information, refer to the instruction on [AlloyDB Omni setup][alloydb-omni].


### CI Platform Setup

Cloud Build is used to run tests against Google Cloud resources in test project: langchain-alloydb-testing.
Each test has a corresponding Cloud Build trigger, see [all triggers][triggers].
These tests are registered as required tests in `.github/sync-repo-settings.yaml`.


#### Trigger Setup

Cloud Build triggers (for Python versions 3.9 to 3.11) were created with the following specs:

```YAML
name: integration-test-pr-py38
description: Run integration tests on PR for Python 3.9
filename: integration.cloudbuild.yaml
github:
  name: langchain-google-alloydb-pg-python
  owner: googleapis
  pullRequest:
    branch: .*
    commentControl: COMMENTS_ENABLED_FOR_EXTERNAL_CONTRIBUTORS_ONLY
ignoredFiles:
  - docs/**
  - .kokoro/**
  - .github/**
  - "*.md"
substitutions:
  _CLUSTER_ID: <ADD_VALUE>
  _DATABASE_ID: <ADD_VALUE>
  _INSTANCE_ID: <ADD_VALUE>
  _REGION: us-central1
  _VERSION: "3.9"
```

Use `gcloud builds triggers import --source=trigger.yaml` to create triggers via the command line

#### Project Setup

1. Create an AlloyDB cluster, instance, and database.
1. Create a private pool for Cloud Build, in order to connect to the VPC network to run tests against a private IP. Learn more at [Run builds in a private pool ](https://cloud.google.com/build/docs/private-pools/run-builds-in-private-pool)
1. Setup Cloud Build triggers (above). Note: Triggers are created in same region as private pool.


#### Run tests with Cloud Build

* Run integration test:

    ```bash
    gcloud builds submit --config integration.cloudbuild.yaml --region us-central1 --substitutions=_INSTANCE_ID=$INSTANCE_ID,_CLUSTER_ID=$CLUSTER_ID,_DATABASE_ID=$DATABASE_ID,_REGION=$REGION
    ```

#### Trigger

To run Cloud Build tests on GitHub from external contributors, ie RenovateBot, comment: `/gcbrun`.


#### Code Coverage
Please make sure your code is fully tested. The Cloud Build integration tests are run with the `pytest-cov` code coverage plugin. They fail for PRs with a code coverage less than the threshold specified in `.coveragerc`.  If your file is inside the main module and should be ignored by code coverage check, add it to the `omit` section of `.coveragerc`.

Check for code coverage report in any Cloud Build integration test log.
Here is a breakdown of the report:
- `Stmts`:  lines of executable code (statements).
- `Miss`: number of lines not covered by tests.
- `Branch`: branches of executable code (e.g an if-else clause may count as 1 statement but 2 branches; test for both conditions to have both branches covered).
- `BrPart`: number of branches not covered by tests.
- `Cover`: average coverage of files.
- `Missing`: lines that are not covered by tests.


[triggers]: https://console.cloud.google.com/cloud-build/triggers?e=13802955&project=langchain-alloydb-testing
[alloydb-omni]: https://cloud.google.com/alloydb/docs/omni/quickstart
