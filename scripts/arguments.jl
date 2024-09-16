using ArgParse

argtable = ArgParseSettings()
@add_arg_table argtable begin
    "--datapath", "-d"
        arg_type = String
        default = "data"
        help = "Path to data folder"

    "--simulationpath", "-s"
        arg_type = String
        default = "simulation/planner"
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

    "--procs"
        arg_type = Int
        default = 0
end