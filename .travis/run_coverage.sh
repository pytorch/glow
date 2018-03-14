#!/bin/bash

ninja all
ninja glow_coverage

COVERAGE_FILE="./glow_coverage/index.html"
if [ ! -f "${COVERAGE_FILE}" ]; then
  echo "ERROR: ${COVERAGE_FILE} not found."
  exit 1
fi

# Travis does not allow using secrets (e.g., AWS credentials) on pull requests
# from a fork. Upload coverage only if secure vars are set.
if [ "${TRAVIS_SECURE_ENV_VARS}" != "false" ]; then
  echo "INFO: Uploading coverage to S3."
  
  BRANCH_NAME="${TRAVIS_BRANCH}"
  COVERAGE_DIR="$(dirname "${COVERAGE_FILE}")"
  UPLOAD_LOCATION="fb-glow-assets/coverage/coverage-${BRANCH_NAME}"

  aws s3 cp "${COVERAGE_DIR}" "s3://${UPLOAD_LOCATION}" --recursive --acl public-read --sse
  echo "INFO: Coverage report for branch '${BRANCH_NAME}': https://fb-glow-assets.s3.amazonaws.com/coverage/coverage-${BRANCH_NAME}/index.html"
else
  echo "WARNING: Coverage cannot be uploaded to s3 for PR from a fork." 
fi
