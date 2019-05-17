#!/bin/bash
ninja all
ninja glow_coverage
COVERAGE_FILE="./glow_coverage/index.html"
if [ ! -f "${COVERAGE_FILE}" ]; then
  echo "ERROR: ${COVERAGE_FILE} not found."
  exit 1
fi

# Upload coverage only on master branch.
if [ "${CIRCLE_BRANCH}" == "master" ]; then
  echo "INFO: Uploading coverage to S3."

  BRANCH_NAME="${CIRCLE_BRANCH}"
  COVERAGE_DIR="$(dirname "${COVERAGE_FILE}")"
  UPLOAD_LOCATION="fb-glow-assets/coverage/coverage-${BRANCH_NAME}"

  aws s3 cp "${COVERAGE_DIR}" "s3://${UPLOAD_LOCATION}" --recursive --acl public-read
  echo "INFO: Coverage report for branch '${BRANCH_NAME}': https://fb-glow-assets.s3.amazonaws.com/coverage/coverage-${BRANCH_NAME}/index.html"
else
  echo "WARNING: Coverage cannot be uploaded to s3 for PR from a fork."
fi
