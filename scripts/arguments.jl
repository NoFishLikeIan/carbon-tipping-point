using ArgParse

argtable = ArgParseSettings()
@add_arg_table argtable begin
    "--datapath", "-d"
        arg_type = String
        default = "data"
        help = "Path to data folder"

    "--simulationpath", "-s"
        arg_type = String
        default = "simulation-small"
        help = "Path to simulation folder"
    
    "--NT"
        arg_type = Int
        default = 21
        help = "Size of grid in temperature"
    
    "--Nm"
        arg_type = Int
        default = 21
        help = "Size of grid in log-CO2e"

    "--verbose" , "-v"
        arg_type = Int
        default = 0
        help = "Verbosity can be set to 0, 1, or 2 and larger."

    "--overwrite"
        action = :store_true

    "--tol"
        arg_type = Float64
        default = 1e-3

    "--dt"
        arg_type = Float64
        default = 1 / 24
    
    "--stopat"
        arg_type = Float64
        default = 0.

    "--cachestep"
        arg_type = Float64
        default = 0.25

    "--threshold"
        arg_type = Float64

    "--damages"
        arg_type = String
    
    "--eis"
        arg_type = Float64
        default = 1.0

    "--rra"
        arg_type = Float64
        default = 10.

    "--withnegative"
        action = :store_true

    "--tau"
        arg_type = Float64
        default = 500.
end

resumeargtable = ArgParseSettings()
@add_arg_table resumeargtable begin
    "filepath"
        arg_type = String
        help = "Path to the file to resume"
    
    "--verbose" , "-v"
        arg_type = Int
        default = 0
        help = "Verbosity can be set to 0, 1, or 2 and larger."
    
    "--stopat"
        arg_type = Float64
        default = 0.

    "--cachestep"
        arg_type = Float64
        default = 0.25
end

ceargstable = ArgParseSettings()
@add_arg_table ceargstable begin
    "--simulationdir", "-s"
        arg_type = String
        default = "simulation-dense"
        help = "Path to simulation folder"

    "--datapath", "-d"
        arg_type = String
        default = "data"
        help = "Path to data folder"
    
    "--calibrationpath"
        arg_type = String
        default = "calibration/"
    
    "--verbose" , "-v"
        arg_type = Int
        default = 0
        help = "Verbosity can be set to 0, 1, or 2 and larger."

    "--threshold"
        arg_type = Float64
        default = 2.0
    
    "--discovery"
        arg_type = Float64
        default = 0.0
    
    "--dt"
        arg_type = Float64
        default = 1 / 24
end

simulateargtable = ArgParseSettings()
@add_arg_table simulateargtable begin
    "--simulationdir", "-s"
        arg_type = String
        default = "simulation-local"
        help = "Path to simulation folder"

    "--datapath", "-d"
        arg_type = String
        default = "data"
        help = "Path to data folder"
    
    "--calibrationpath"
        arg_type = String
        default = "calibration/"
    
    "--verbose" , "-v"
        arg_type = Int
        default = 0
        help = "Verbosity can be set to 0, 1, or 2 and larger."

    "--threshold"
        arg_type = Float64
        default = 2.0
    
    "--discovery"
        arg_type = Float64
        default = 0.0

    "--horizon"
        arg_type = Float64
        default = 80.0
        
    "--trajectories"
        arg_type = Int64
        default = 100
end