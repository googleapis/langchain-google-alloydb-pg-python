# Changelog

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
