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
    
    "-N"
        arg_type = Int
        default = 51
        help = "Size of grid"

    "--verbose" , "-v"
        arg_type = Int
        default = 0
        help = "Verbosity can be set to 0, 1, or 2 and larger."

    "--overwrite"
        action = :store_true

    "--tol"
        arg_type = Float64
        default = 1e-5
    
    "--stopat"
        arg_type = Float64
        default = 0.

    "--cachestep"
        arg_type = Float64
        default = 0.25

    "--procs", "-p"
        arg_type = Int
        default = 0

    "--threshold"
        arg_type = Float64
        default = 1.5

    "--leveldamages"
        action = :store_true
    
    "--eis"
        arg_type = Float64
        default = 0.75

    "--rra"
        arg_type = Float64
        default = 10.

    "--allownegative"
        action = :store_true
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