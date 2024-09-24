using ArgParse

argtable = ArgParseSettings()
@add_arg_table argtable begin
    "simulationpath"
        arg_type = String
        required = true # Commented for REPL.
        help = "Path to simulation file"

    "--datapath"
        arg_type = String
        default = "data"

    "--experimentpath"
        arg_type = String
        default = "experiments"
        help = "Path to experiment file"

    "--verbose" , "-v"
        arg_type = Int
        default = 0
        help = "Verbosity can be set to 0, 1, or 2 and larger."
    
    "--trajectories"
        arg_type = Int
        default = 100
end