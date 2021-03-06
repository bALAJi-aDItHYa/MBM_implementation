Requirements:
1) Matlab version R2019b
	-  Image Processing toolbox
	-  Parallel processing toolbox
	-  Pdollar's Toolbox for edge detection (copy source from github)

2) Steps:
	i) Clone the pdollar edges repository to get the scripts for edge detection

	ii) Clone the pdollar toolbox repo to get the toolbox (note: requires that image processing toolbox be already installed in Matlab)

	iii) Important folders: -> img - contains the RGB images
			      gt    - contains the edge maps (ground truth as provided by BIPED dataest)
			      pred - contains the predicted edge maps from the inference run on PyTorch
			      edge_nms - *empty folder - soon will be filled with thinned edge maps after post processing on Matlab*

3) Main Files for post processing:
	1) main_eval.m - 		
			a) Adds the necessary paths to the pdollar edge detection toolbox
			b) Performs the nms (Non-maximum Suppression) operation and edge thinning and stores the .png files in edge_nms folder
			c) Function call to edgesEvalDir_x() which returns the necessary error metric values
	
	2) edgesEvalDir_x.m - 
			Input: nms_pred_dir, gtDir -> the path to directory containing the post processed predicted edge maps from the DexiNed model and the ground truth edge maps respectively
			Output: ODS, OIS, AP

			Executed to compute the ODS, OIS and AP error metric values and returns them.

	As Pdollar's toolbox focuses on ground-truth being .mat files, some files had to modified to account for the .png format of the ground-truth images.

Source code:
1) https://github.com/xavysp/DexiNed.git
2) https://github.com/pdollar/edges.git
3) https://github.com/pdollar/toolbox.git
