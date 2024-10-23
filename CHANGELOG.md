# Changelog

## [0.8.0](https://github.com/googleapis/langchain-google-alloydb-pg-python/compare/v0.7.0...v0.8.0) (2024-10-23)


### Features

* Add GCS URI support for image embedding ([#247](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/247)) ([2f4ee36](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/2f4ee3644675ca7fcb05ed779adeb2672b73f7c7))
* Enable model endpoint management for embeddings ([#233](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/233)) ([247b9cf](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/247b9cfdbc758c68b976183831cdff7d41f11848))
* Functionality to help users migrate data from PGVector to AlloyDB ([#235](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/235)) ([013e2a1](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/013e2a179e9d4ec0d2ede111f187a98affa57d61))
* Remove support for Python 3.8 ([#237](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/237)) ([d0a8801](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/d0a8801e3c569647cc9392ea25a5d22b4aff889f))


### Documentation

* Added pg to alloy migration guide ([#249](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/249)) ([3ee3601](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/3ee360128c4687311a7c20a38ca2a1b0f9861c1b))

## [0.7.0](https://github.com/googleapis/langchain-google-alloydb-pg-python/compare/v0.6.0...v0.7.0) (2024-09-17)


### ⚠ BREAKING CHANGES

* support async and sync versions of indexing methods
* remove _aexecute(), _execute(), _afetch(), and _fetch() methods

### Features

* Add from_engine_args method ([f06834d](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/f06834d67b3219d0981dbad191f0c6d89dc8aa03))
* Add multi-modal support vector store ([#207](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/207)) ([8d259ba](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/8d259ba0c98476b312e7a7201fa86f9316919ab6))
* Add support for custom schema names ([#210](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/210)) ([f148a7e](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/f148a7e7ebd58afc18ee32f42bfe980a5fde9907))
* Add support for sync from_engine ([f06834d](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/f06834d67b3219d0981dbad191f0c6d89dc8aa03))
* Allow non-uuid data types for vectorstore primary key ([#226](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/226)) ([e6dc991](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/e6dc9918afa42086cc975056bb23862d4bb7017e))
* Made add_embeddings method public ([#228](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/228)) ([433b92f](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/433b92f4a2c62ccc2244c95feba9e1e0ec9620ea))
* Refactor to support both async and sync usage ([f06834d](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/f06834d67b3219d0981dbad191f0c6d89dc8aa03))


### Bug Fixes

* Fix documentation comments for library methods. ([#219](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/219)) ([5b923b2](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/5b923b299fba54e3be36363f070fec7264444aea))
* Replacing cosine_similarity and maximal_marginal_relevance methods ([#218](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/218)) ([d827ccc](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/d827ccce0ba3b032e636b26e9ea5c5a5d3e151f6))
* Support async and sync versions of indexing methods ([f06834d](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/f06834d67b3219d0981dbad191f0c6d89dc8aa03))


### Documentation

* Added example for chat message history with omni ([#191](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/191)) ([2e6809f](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/2e6809f0ae0063307d49529e784e65a4929e9c85))
* Add Migration samples for migrating from vector store to AlloyDB ([#230](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/230)) ([9fd9308](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/9fd93081aa8307d9190f7f30ef6746e0e9aeedbe))

## [0.6.0](https://github.com/googleapis/langchain-google-alloydb-pg-python/compare/v0.5.0...v0.6.0) (2024-09-03)


### ⚠ BREAKING CHANGES

* Update `postgres_ann` to `alloydb_scann` ([#212](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/212))

### Features

* Add ScaNN index query options ([#198](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/198)) ([0813867](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/081386779ef80e419cb734d21673a8feb195a271))


### Bug Fixes

* Fix `QueryOptions` not applied to similarity search bug ([#203](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/203)) ([df206d0](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/df206d090d3b1cc83770d0052095362002ff377c))
* Update `postgres_ann` to `alloydb_scann` ([#212](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/212)) ([783dcad](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/783dcad96b3cf0ae2ac11af826b71dff67a47f1f))


### Documentation

* Add index choosing guide ([#200](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/200)) ([dd91d54](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/dd91d54f0bfc2942704537b062862b038f9f3891))
* Added info on how to modify existing libraries ([#205](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/205)) ([202b086](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/202b086678bcd55ccc2e69592d8db7fbfe4f5acb))
* Added vector store initialization from documents ([#196](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/196)) ([823821c](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/823821c647b2e0fd6a7a44d4a2ed6d11ffb1037a))
* Create functions need pre-existing tables ([#202](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/202)) ([a2e6c0a](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/a2e6c0a1d1ef2887d3fc9edac7cd62a0864f5a17))
* Rename index guide ([#201](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/201)) ([2436c5b](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/2436c5b2120d69938644e20181559aa62b95fc84))
* Update index tuning samples to include advanced indexes ([#197](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/197)) ([da52bc4](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/da52bc421da1ca8cad78184566426b22941b09af))

## [0.5.0](https://github.com/googleapis/langchain-google-alloydb-pg-python/compare/v0.4.1...v0.5.0) (2024-07-23)


### Features

* Add similarity search score threshold select function ([#181](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/181)) ([ab0a574](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/ab0a5746b21971c6ef8bc24f45d620d1a5925176))
* Add table name to default index name ([#182](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/182)) ([5e746ce](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/5e746ce037a0040479eab75f1aa7147b949d487a))
* Add tests to reach 90% coverage ([#193](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/193)) ([390d945](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/390d945e2442cf47f6d7bb6c9d7cfb1746df63d3))
* Added example for document saver ([#185](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/185)) ([e0aab6f](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/e0aab6f9a297163fe009048c654e679c0de73e3b))
* Added guide for using existing table as vector store ([#170](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/170)) ([adefe60](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/adefe6019a3e875ebef2b5ed8866ecd31db52e69))
* Auto-generate IDs upon adding embeddings ([#184](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/184)) ([3e5a381](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/3e5a381be78e09b95be031eb47d59d64fa2c39a8))
* Remove `alloydb_` prefix from module names ([#173](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/173)) ([9c80bf3](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/9c80bf3c655ae7b32bb384c4ed850c9aa931ebca))
* Support IAM account override ([#188](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/188)) ([5c3d5ea](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/5c3d5ea6dc4acbf499b1ed34649154035ef9a5b1))


### Bug Fixes

* Add `key` to engine constructor ([#179](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/179)) ([497fec4](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/497fec489ab32a6dda1b19e3e686c7b872461847))
* Remove `langchain-community` dependency ([#178](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/178)) ([96fd44d](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/96fd44dde257d3dd96e841e2e433691ad4ff903e))
* Rename inner product distance search function to inner_product ([#195](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/195)) ([7eee4e6](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/7eee4e6ac4037de44e9fccc136ef566d6d9c8c90))


### Documentation

* Add docstring to all methods ([#187](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/187)) ([518581e](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/518581e68cb552a07207c9f794de03a03ed97ce7))

## [0.4.1](https://github.com/googleapis/langchain-google-alloydb-pg-python/compare/v0.4.0...v0.4.1) (2024-06-27)


### Bug Fixes

* Change IVFFlat `list` default to 100 ([#165](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/165)) ([f4d9b42](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/f4d9b4204a465ef10d5cc23d1fd6ce349fb22a21))
* Use lazy refresh for AlloyDB Connector ([#162](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/162)) ([3a9c6ac](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/3a9c6acb2e368a099267c9e45acb86ef38195f7a))

## [0.4.0](https://github.com/googleapis/langchain-google-alloydb-pg-python/compare/v0.3.0...v0.4.0) (2024-06-18)


### Features

* Add IVF and ScaNN Indexes ([#139](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/139)) ([f9d70f1](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/f9d70f16aa699012d79068a0870f475d35b7ad94))

## [0.3.0](https://github.com/googleapis/langchain-google-alloydb-pg-python/compare/v0.2.2...v0.3.0) (2024-05-30)


### Features

* Support LangChain v0.2 ([#138](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/138)) ([2d2d119](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/2d2d1190975da805d43e9eae843f504dc323d82c))

## [0.2.2](https://github.com/googleapis/langchain-google-alloydb-pg-python/compare/v0.2.1...v0.2.2) (2024-04-30)


### Bug Fixes

* Missing quote from table name ([#118](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/118)) ([b1d80f2](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/b1d80f28411f64cbe7437d960715dc4a8f8d0b8e))

## [0.2.1](https://github.com/googleapis/langchain-google-alloydb-pg-python/compare/v0.2.0...v0.2.1) (2024-04-30)


### Bug Fixes

* Delete for vector store document ([#71](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/71)) ([34885d0](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/34885d0df9c7ba03ebbdac34e95a050cf6f9f8a7))
* Update required dependencies to use SQLAlchemy[asyncio] ([#116](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/116)) ([1c51033](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/1c51033961bb9840ec389da2ef1dc2237dd5e63f))

## [0.2.0](https://github.com/googleapis/langchain-google-alloydb-pg-python/compare/v0.1.0...v0.2.0) (2024-03-25)


### Features

* **ci:** Test against multiple python versions ([#57](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/57)) ([c398581](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/c39858126006835fc1ff680ee4e6199d3f8d12ca))


### Bug Fixes

* Sql statement for non-nullable columns ([#59](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/59)) ([84c9d26](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/84c9d26d00b58220c082c99a91e9d5308760c605))


### Documentation

* Add Demo Langchain Application. ([#42](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/42)) ([9fd2aa7](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/9fd2aa7cecaa6d1a0010f8ced5f518096a167388))
* Add github links ([#51](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/51)) ([0f86e02](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/0f86e022fd2f698da00b7098d151ac22b5ca2d5e))
* Correcting vectorstore import ([#64](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/64)) ([1309da6](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/1309da67b0a7d431a15593bf5bc414f54aeb5290))
* Update CODEOWNERS ([#45](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/45)) ([f5e0e01](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/f5e0e0150962ab352da27317d67fe0710108444d))
* Update langchain_quick_start.ipynb ([#47](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/47)) ([026cd63](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/026cd6324fac77daab3566ea310adcac5fbaa8ac))
* Update langchain_quick_start.ipynb ([#52](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/52)) ([8108c02](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/8108c02fe66a66b1ba11c91c452364794fdc33d7))
* Update langchain_quick_start.ipynb ([#53](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/53)) ([31b90c0](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/31b90c0d66c495e623e629f92bad0d30bd2ad33e))
* Update langchain_quick_start.ipynb ([#54](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/54)) ([41ccee8](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/41ccee89d53bf4b7f073da6fd9eb4e0e0bc74920))
* Update langchain_quick_start.ipynb ([#55](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/55)) ([56cdda2](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/56cdda22cdac196f472bab2e85c37e324639588e))
* Update README.md ([#46](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/46)) ([e488bbd](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/e488bbd0b361203763006b83af9eeb6741438221))
* Update tutorials and add GitHub links ([#49](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/49)) ([96f4515](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/96f45152efaaa1dcc7d0776d9234ebeac50992e5))

## 0.1.0 (2024-02-28)


### Features

* Add AlloyDB chatmessagehistory ([#11](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/11)) ([83cabec](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/83cabec291aef67c5e3fd6dd32c683092484b934))
* Add alloydb vectorstore ([#9](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/9)) ([863c320](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/863c3203dfb73bded159c60b60aba6534f94b7f4))
* Add AlloyDBEngine ([#7](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/7)) ([bf4de16](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/bf4de163ffc6ccc4f6852eaba17c0fb15d2b4c37))
* Add document loader ([#18](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/18)) ([7eb6ea5](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/7eb6ea5eb987965f4e2a900239e8016b5dda8925))
* Add document saver ([#19](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/19)) ([7762d64](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/7762d64efce21e7a27f0b2e917290a4e1272c6f1))


### Documentation

* Add docs and integration tests for alloydbchatmessagehistory ([#17](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/17)) ([1ac8f0d](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/1ac8f0d93adbd8e6757dde186157944457265095))
* Add Document Loader docs ([#23](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/23)) ([d49f030](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/d49f030362f4642ace068e2f8ce1d1f3d94e569c))
* Add Vectorstore docs ([#22](https://github.com/googleapis/langchain-google-alloydb-pg-python/issues/22)) ([a88b8b1](https://github.com/googleapis/langchain-google-alloydb-pg-python/commit/a88b8b163b776c885a8b6ce68119d5d75d3bb7c0))
