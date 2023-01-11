def compile_all_fish(root_path,name_of_data):
    print('compilation start')
    #For accessing all_bout_pandas
    df = pd.DataFrame()

    for fish_path in tqdm(root_path.glob("*")):

        if not fish_path.is_dir():
            continue
        if len(h5py.File(fish_path / f"{fish_path.name}.hdf5").keys()) == 1:
            print(f"File {fish_path / f'{fish_path.name}.hdf5'} is empty.")
        else:
            df_for_concat = pd.read_hdf(fish_path / f"{fish_path.name}.hdf5")

            df = pd.concat((df, df_for_concat))
    df.sort_index(inplace=True)
    df.sort_values(by=['fish_name', 'trial'], inplace=True, ascending=True)
    print('compilation done')
    print('labeling start')
    df = label_bouts(df, name_of_data,bout_angle_threshold = 2)
    print('binning start')
    bin_data(df,name_of_data, bin_size = 1)
    print('all done, well done darling')

root_path = Path(r"C:\Users\ag-bahl\Desktop\all_data_gray_8_directions")
name_of_data = "gray_8_directions"
compile_all_fish(root_path,name_of_data)









