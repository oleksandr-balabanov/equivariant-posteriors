#!/bin/sh
set -x
set -e
PG_PATH=$(pwd)/pg_database
export LC_CTYPE=en_US.UTF-8
export LANG=en_US.UTF-8
mkdir -p $PG_PATH
echo postgres > /tmp/pg
$(readlink -f $(which initdb)) -L $POSTGRES/share/postgresql --pwfile /tmp/pg -U postgres $PG_PATH || true
rm -f $PG_PATH/pg_hba.conf
cat <<EOM >$PG_PATH/pg_hba.conf
local   all             all                                     md5
host    all             all             0.0.0.0/0               md5
host    all             all             ::1/128                 md5
EOM
$(readlink -f $(which postgres)) -D $PG_PATH -k $PG_PATH -i
