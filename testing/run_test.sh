#!/bin/bash

red=$(tput setaf 1)
green=$(tput setaf 2)
reset=$(tput sgr0)

test_files=$(ls ./bin/test_*)

echo "${green}Run testing${reset}"

for file in $test_files; do

  $file;

  if [[ $? != 0 ]]; then
    echo "${red}ERROR${reset}"
    exit 1;
  fi

done

echo "${green}PASSED${reset}"
