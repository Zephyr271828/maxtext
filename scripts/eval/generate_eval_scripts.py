import os
from textwrap import dedent

# Example usage
if __name__ == "__main__":

    for model_name in ["llama3.1-4b-depth", "llama3.1-4b-width"]:
        for num_steps in [12500, 62500]:
            generate_script(
                model_name=model_name,
                num_steps=num_steps,
                # load_parameters_path="model_ckpts/llama3.1-4b-depth-orbax/0/items",
                # load_parameters_path=args.load_parameters_path,
                # output_path=args.output_path,
            )
            
    for model_name in ["llama3.1-8b"]:
        for num_steps in [50000, 37500, 25000, 12500]:
            generate_script(
                model_name=model_name,
                num_steps=num_steps,
                # load_parameters_path="model_ckpts/llama3.1-4b-depth-orbax/0/items",
                # load_parameters_path=args.load_parameters_path,
                # output_path=args.output_path,
            )
            
    for model_name in [
        "llama3.1-440m", "llama3.1-1b", 
        "llama3.1-1.5b-depth", "llama3.1-2b-depth", "llama3.1-3b-depth",
        "llama3.1-2b-width", "llama3.1-3b-width",
    ]:
        for num_steps in [12500, 62500]:
            generate_script(
                model_name=model_name,
                num_steps=num_steps,
                # load_parameters_path="model_ckpts/llama3.1-4b-depth-orbax/0/items",
                # load_parameters_path=args.load_parameters_path,
                # output_path=args.output_path,
            )
            
    for load_path, model_name in zip(
        ["model_ckpts/maxtext/llama3.1_minitron_depth_hf/0/items", "model_ckpts/maxtext/llama3.1_minitron_width_hf/0/items"],
        ["llama3.1-4b-depth", "llama3.1-4b-width"]
    ):
        for num_steps in [12500]:
            generate_script(
                model_name=model_name,
                num_steps=num_steps,
                load_parameters_path=load_path,
                pretrain_tokens="L200",
                # load_parameters_path="model_ckpts/llama3.1-4b-depth-orbax/0/items",
                # load_parameters_path=args.load_parameters_path,
                # output_path=args.output_path,
            )
            
    for load_path, model_name in zip(
        ["model_ckpts/maxtext/llama3.1-4b-depth-orbax/checkpoints/0/items", "model_ckpts/maxtext/llama3.1-4b-width-orbax/checkpoints/0/items"],
        ["llama3.1-4b-depth", "llama3.1-4b-width"]
    ):
        for num_steps in [12500]:
            generate_script(
                model_name=model_name,
                num_steps=num_steps,
                load_parameters_path=load_path,
                pretrain_tokens="HF",
                # load_parameters_path="model_ckpts/llama3.1-4b-depth-orbax/0/items",
                # load_parameters_path=args.load_parameters_path,
                # output_path=args.output_path,
            )
                    
                    
    for load_path, model_name in zip(
        ["model_ckpts/maxtext/llama3.1-1.5b-depth-minitron/checkpoints/0/items", "model_ckpts/maxtext/llama3.1-2b-depth-minitron/checkpoints/0/items", "model_ckpts/maxtext/llama3.1-3b-depth-minitron/checkpoints/0/items"],
        ["llama3.1-1.5b-depth", "llama3.1-2b-depth", "llama3.1-3b-depth"]
    ):
        for num_steps in [12500]:
            generate_script(
                model_name=model_name,
                num_steps=num_steps,
                load_parameters_path=load_path,
                pretrain_tokens="L200",
                # load_parameters_path="model_ckpts/llama3.1-4b-depth-orbax/0/items",
                # load_parameters_path=args.load_parameters_path,
                # output_path=args.output_path,
            )
            
    for load_path, model_name in zip(
        [
            "model_ckpts/llama3-8b-l200_width_task_wikitext_hidden_size_1792_ffn_hidden_size_5632_calib_size_128_seqlen_8192/checkpoints/0/items", 
            "model_ckpts/llama3-8b-l200_width_task_wikitext_hidden_size_2432_ffn_hidden_size_6144_calib_size_128_seqlen_8192/checkpoints/0/items"
        ],
        [   
            "llama3.1-2b-width", 
            "llama3.1-3b-width"
        ]
    ):
        for num_steps in [12500]:
            generate_script(
                model_name=model_name,
                num_steps=num_steps,
                load_parameters_path=load_path,
                pretrain_tokens="L200",
                # load_parameters_path="model_ckpts/llama3.1-4b-depth-orbax/0/items",
                # load_parameters_path=args.load_parameters_path,
                # output_path=args.output_path,
            )
                    
    for load_path in [
        "model_ckpts/maxtext/llama3.1_8b_L200_unstructured_0.5/checkpoints/0/items",
        "model_ckpts/maxtext/llama3.1_8b_L200_4:8_0.5/checkpoints/0/items", 
        "model_ckpts/maxtext/llama3.1_8b_L200_2:4_0.5/checkpoints/0/items",
        
        "model_ckpts/maxtext/llama3.1-8b_l200_sparsegpt_unstructured_0.5/checkpoints/0/items",
        "model_ckpts/maxtext/llama3.1-8b_l200_sparsegpt_2:4_0.5/checkpoints/0/items",
       
    ]:
        generate_script(
            model_name="llama3.1-8b",
            num_steps=12500,
            load_parameters_path=load_path,
            sparse_model_training=True,
            pretrain_tokens="L200",
            # load_parameters_path="model_ckpts/llama3.1-4b-depth-orbax/0/items",
            # load_parameters_path=args.load_parameters_path,
            # output_path=args.output_path,
        )
    
    for load_path in [
        "model_ckpts/maxtext/llama3.1_8b_L200_unstructured_0.5_reinit/checkpoints/0/items",
        # "model_ckpts/maxtext/llama3.1_8b_L200_4:8_0.5_reinit/checkpoints/0/items",
        "model_ckpts/maxtext/llama3.1_8b_L200_2:4_0.5_reinit/checkpoints/0/items",
        "model_ckpts/maxtext/llama3.1-8b_l200_sparsegpt_unstructured_0.5_reinit/checkpoints/0/items",
        "model_ckpts/maxtext/llama3.1-8b_l200_sparsegpt_2:4_0.5_reinit/checkpoints/0/items",
    ]:
        for num_steps in [12500, 62500]:
            generate_script(
                model_name="llama3.1-8b",
                num_steps=num_steps,
                load_parameters_path=load_path,
                sparse_model_training=True,
                # pretrain_tokens="L200",
                # load_parameters_path="model_ckpts/llama3.1-4b-depth-orbax/0/items",
                # load_parameters_path=args.load_parameters_path,
                # output_path=args.output_path,
            )
        
    for model_name in ["llama2-1.3b", "llama2-2.7b"]:
        for num_steps in [12500, 62500]:
            generate_script(
                model_name=model_name,
                num_steps=num_steps,
                seq_len=4096,
                data_files="/home/zephyr/gcs-bucket/datasets/dclm/llama2_array_record_with_special_tokens_64/*.array_record",
                # load_parameters_path="model_ckpts/llama3.1-4b-depth-orbax/0/items",
                # load_parameters_path=args.load_parameters_path,
                # output_path=args.output_path,
            )
            
    for model_name in ["llama2-7b"]:
        for num_steps in [50000]:
            generate_script(
                model_name=model_name,
                num_steps=num_steps,
                seq_len=4096,
                data_files="/home/zephyr/gcs-bucket/datasets/dclm/llama2_array_record_with_special_tokens_64/*.array_record",
                # load_parameters_path="model_ckpts/llama3.1-4b-depth-orbax/0/items",
                # load_parameters_path=args.load_parameters_path,
                # output_path=args.output_path,
            )