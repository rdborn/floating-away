#!/usr/bin/env bash

export AZURE_STORAGE_ACCOUNT=prediction
export AZURE_STORAGE_ACCESS_KEY=5tsKK68CZWDXP5GHPhLGcplCMq8x2fuWEEjgVFYHynqsI0qNvcd3ZFKWaWxZACr0WLVbfB/2ty00piibjTOZ4g==

gem install azure-storage -v 0.14.0.preview --pre
gem install activesupport

ruby "${BASH_SOURCE%/*}/standalone.rb"
