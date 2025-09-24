

declare -A config=(
	['rpath']="WRITE DOWN THE PATH TO YOUR DATA"
	['tf']='costco'
	['dataset']="DATANAME"
	['bs']="1024"
	# MODEL CONFIGURATION
	['epoch']=10000
	['device']="cuda:0"
)

declare -A exp_config=(

    # MODEL CONFIGURATION
    ['wd']="0.0001 0.00001"
    ['lr']="0.0001 0.001"
    ['nc']="8 16 32 64 128"
    ['rank']="8 16 32 64 128"
)



# Fixed arguments
fixed_args=()
for key in ${!config[@]}; do
	fixed_args+=("--${key} ${config[${key}]}")
done

#Arguments for experiments
for i in 1 2 3; do
for rank in ${exp_config['rank']}; do
	for lr in ${exp_config['lr']}; do
		for wd in ${exp_config['wd']}; do
			for nc in ${exp_config['nc']}; do
			var_args=(--rank $rank --lr $lr --wd $wd --nc $nc)
			total_args=(${fixed_args[@]} ${var_args[@]})
			echo ${total_args[@]}
			TF_CPP_MIN_LOG_LEVEL=2 python main.py "${total_args[@]}"
		done
	done
done
done
done

