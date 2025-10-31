CREATE TYPE bats_enum AS ENUM ('left', 'right', 'switch');
CREATE TYPE throws_enum AS ENUM ('left', 'right');

CREATE TABLE IF NOT EXISTS 'Player' (
    /* TODO: add player table fields here */
    player_id varchar(255) primary key,
    last_name varchar(255),
    first_name varchar(255),
    bats bats_enum,
    throws throws_enum,
    birth_date date,
    height int, /* in inches */
    weight int /* in lbs */
);

CREATE TABLE IF NOT EXISTS 'Contract' (
    /* TODO: add contract table fields here */
);

CREATE TABLE IF NOT EXISTS 'HittingStats' (
    /* TODO: add hitting stats table fields here */
);

CREATE TABLE IF NOT EXISTS 'PitchingStats' (
    /* TODO: add pitching stats table fields here */
);