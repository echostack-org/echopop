-- =================================================================
--  Database Seed File
--  Generated from input_files document.
-- =================================================================

-- Drop existing objects --

DROP TABLE IF EXISTS echopop_catch CASCADE;
DROP TABLE IF EXISTS echopop_fish CASCADE;
DROP TYPE IF EXISTS sex_enum;

CREATE TYPE sex_enum AS ENUM (
    'male',
    'female',
    'unsexed'
);

-- Create Main Data Tables --

CREATE TABLE echopop_fish (
    ship INTEGER NOT NULL,
    survey INTEGER NOT NULL,
    haul_num INTEGER NOT NULL,
    species_code INTEGER NOT NULL,

    sex sex_enum NOT NULL DEFAULT 'unsexed',

    -- cm
    length DECIMAL(10, 2) CHECK (length > 0),

    -- kg
    weight DECIMAL(10, 3) CHECK (weight > 0),

    -- years
    age DECIMAL(5, 1) CHECK (age >= 0)
);

CREATE TABLE echopop_catch (
    ship INTEGER NOT NULL,
    survey INTEGER NOT NULL,
    haul_num INTEGER NOT NULL,
    species_code INTEGER NOT NULL,

    -- kg
    weight_in_haul DECIMAL(10, 3) NOT NULL CHECK (weight_in_haul >= 0),

    gear VARCHAR(50),
    net_num INTEGER,

    -- Ensure only one weight entry per haul/species
    UNIQUE(ship, survey, haul_num, species_code)
);

-- Insert Data --

INSERT INTO echopop_fish (ship, survey, haul_num, species_code, sex, length, weight, age) VALUES
(101, 2024, 1, 22500, 'male', 30.5, 0.450, 4.0),
(101, 2024, 1, 22500, 'male', 31.0, 0.465, 4.0),
(101, 2024, 1, 22500, 'unsexed', 20.0, 0.2, 2.0),
(101, 2024, 1, 22500, 'female', 32.0, 0.510, 5.0),
(101, 2024, 1, 22500, 'unsexed', 15.2, NULL, 1.0),      -- NULL weight
(101, 2024, 1, 206, 'female', 25.0, 0.300, 3.0),
(101, 2024, 1, 206, 'female', 26.5, 0.320, 3.0),
(101, 2024, 2, 22500, 'male', 40.0, 0.600, 6.0),
(101, 2024, 2, 22500, 'female', 42.5, 0.650, 7.0),
(101, 2024, 2, 22500, 'unsexed', NULL, NULL, NULL), -- All info missing
(102, 2024, 1, 150, 'female', 45.0, 1.200, 10.0),
(102, 2024, 1, 150, 'male', 40.0, 0.950, 8.0),
(101, 2024, 1, 22500, 'male', 30.5, NULL, NULL),
(101, 2024, 1, 22500, 'male', 31.0, NULL, NULL),
(101, 2024, 1, 22500, 'unsexed', 20.0, NULL, NULL),
(101, 2024, 1, 22500, 'female', 32.0, NULL, NULL),
(101, 2024, 1, 22500, 'female', 31.0, NULL, NULL),
(101, 2025, 1, 206, 'male', 35.0, 0.500, 5.0);

INSERT INTO echopop_catch (ship, survey, haul_num, species_code, weight_in_haul, gear, net_num) VALUES
(101, 2024, 1, 22500, 120.500, 'Aleutian Wing Trawl', 5880),
(101, 2024, 1, 206, 75.200, 'Aleutian Wing Trawl', 5880),
(101, 2024, 1, 150, 50.000, 'Aleutian Wing Trawl', 5880),
(101, 2024, 2, 22500, 250.000, 'Aleutian Wing Trawl', 5594),
(101, 2024, 3, 22500, 230.000, 'Aleutian Wing Trawl', 5594),
(102, 2024, 1, 150, 50.000, 'Aleutian Wing Trawl', 5594),
(102, 2024, 2, 22500, 40.000, NULL, NULL),
(101, 2025, 1, 206, 90.000, 'Aleutian Wing Trawl', NULL);
