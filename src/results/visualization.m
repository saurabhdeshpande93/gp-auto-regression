(* ::Package:: *)

(* ::Input:: *)
(*<<AceFEM`;*)
(**)
(*(* Add on libraries to import external mesh to AceFEM *)*)
(*PacletFind["FEMAddOns"];*)
(*Needs["FEMAddOns`"];*)
(*Get["ImportMesh`"];*)
(*mesh = Import["liver.msh","ElementMesh"];*)
(**)
(*(*Initialise the input data for the AceFEM*)*)
(*SMTInputData[]; *)
(*SMTAddDomain[{"\[CapitalOmega]",{"ML:","SE","D3","O1","DF","HY","O1","D","NeoHookeWA"},{"E *"->5000,"\[Nu] *"->0.45,"\[Rho] *"->1000}}]; (*define domain*)*)
(*SMTAddMesh[mesh,"\[CapitalOmega]"];  (*Assign mesh to above defined domain*)*)
(*SMTAnalysis[];(*Initialise the analysis*)*)
(*mrest = SMTShowMesh["BoundaryConditions"->False,"DeformedMesh"->False,"Mesh"->Gray,"FillElements"->False,"ImageSize"->450, ViewPoint->{-0.44174442243945883`,1.9345565465705503`,-0.3810953769444234`},ViewVertical->{-0.015039029778453047`,0.2397723166056823`,0.9707126576762389`}](*Show the undeformed mesh*)*)
(**)
(*(*Get the command-line arguments*)*)
(*args=$ScriptCommandLine;*)
(**)
(*(* Otherwise provide the test number manually : Default value *)*)
(*testnumber=276;  (*Choose the test testnumber to visualise*)*)
(**)
(*(*Check if an testnumber is provided, otherwise use a default value*)*)
(*testnumber=If[Length[args]>1,ToExpression[args[[2]]],testnumber];*)
(**)
(*(* Import the csv file *)*)
(*filePath=StringTemplate["test_`no`/t`no`.csv"][<|"no"->testnumber|>];*)
(*input=Import[filePath];*)
(**)
(*(* GP + Autoencoder framework mean prediction ('full_disps' obtained using predict.py) *)*)
(*nn= input[[All,1]] ;*)
(*listdisp = Partition[nn ,3]; (* reshape *)*)
(*dispnn =  (Norm/@listdisp); (* take row wise norm to get nodal values instead of dof values *)*)
(**)
(*(* True FEM solution *)*)
(*fem = input[[All,2]] ;*)
(*listfem = Partition[fem ,3];*)
(*dispfem =  (Norm/@listfem);*)
(**)
(*(* Error of the framework (e_f) *)*)
(*error = input[[All,3]];*)
(*list = Partition[error,3];*)
(*nodeerror = (Norm/@list); *)
(**)
(*(* GP + Autoencoder framework uncertainty prediction (full_sigmas) *)*)
(*uncertain = input[[All,4]];*)
(*listuncertain = Partition[uncertain,3];*)
(*nodeuncertain = (Norm/@listuncertain);*)
(**)
(*(* GP component error (e_gp) *)*)
(*gperror = input[[All,5]];*)
(*listgperror = Partition[gperror,3];*)
(*nodegperror = (Norm/@listgperror);*)
(**)
(*(* Reconstruction error (e_r) *)*)
(*autoerror = input[[All,6]];*)
(*listautoerror = Partition[autoerror,3];*)
(*nodeautoerror = (Norm/@listautoerror);*)
(*Print[" The FEM solution is depicted using a red mesh, whereas the framework solution is shown with a blue mesh. "]*)
(*(*Plot FEM mesh: Red Color*)*)
(*SMTNodeData["at",Partition[fem ,3]]; (* Assign FEM solution to node data, this will give deformed mesh obtained by FEM.*)*)
(*mfem = SMTShowMesh["BoundaryConditions"->False,"DeformedMesh"->True,"Mesh"->Red,"FillElements"->False,Lighting->{{"Ambient",White}},"ImageSize"->100,ViewPoint->{-0.44174442243945883`,1.9345565465705503`,-0.3810953769444234`},ViewVertical->{-0.015039029778453047`,0.2397723166056823`,0.9707126576762389`}];*)
(**)
(*(*Plot FEM displacement contours*)*)
(*mdispfem = SMTShowMesh["BoundaryConditions"->False,"DeformedMesh"->True,"Mesh"->Black,"FillElements"->False, "Field"-> dispfem,"ImageSize"->400,Lighting->"Neutral","Label"-> "FEM solution",ViewPoint->{-0.44174442243945883`,1.9345565465705503`,-0.3810953769444234`},ViewVertical->{-0.015039029778453047`,0.2397723166056823`,0.9707126576762389`},"Contour"->{0.679,4.81,5}] ;*)
(**)
(*(*Assign GP+Auto predictions. Blue mesh*) *)
(*SMTNodeData["at",Partition[nn ,3]]; (* Assign framework predictions to node data.*)*)
(*mnn = SMTShowMesh["BoundaryConditions"->False,"DeformedMesh"->True,"Mesh"->Blue,"FillElements"->False,Lighting->{{"Ambient",White}},"ImageSize"->100,ViewPoint->{-0.44174442243945883`,1.9345565465705503`,-0.3810953769444234`},ViewVertical->{-0.015039029778453047`,0.2397723166056823`,0.9707126576762389`}];*)
(**)
(*(*Plot nodal displacement*)*)
(*mdispnn= SMTShowMesh["BoundaryConditions"->False,"DeformedMesh"->True,"Mesh"->Black,"FillElements"->False, "Field"-> dispnn,"ImageSize"->400,Lighting->"Neutral","Label"-> "GP+NN mean prediction",ViewPoint->{-0.44174442243945883`,1.9345565465705503`,-0.3810953769444234`},ViewVertical->{-0.015039029778453047`,0.2397723166056823`,0.9707126576762389`},"Contour"->{0.679,4.81,5}] ;*)
(**)
(*(*Plot prediction error*)*)
(*merror = SMTShowMesh["BoundaryConditions"->False,"DeformedMesh"->True,"Mesh"->Black,"FillElements"->False, "Field"-> nodeerror,"ImageSize"->400,Lighting->"Neutral","Label"-> "e_framework",ViewPoint->{-0.44174442243945883`,1.9345565465705503`,-0.3810953769444234`},ViewVertical->{-0.015039029778453047`,0.2397723166056823`,0.9707126576762389`},"Contour"->True] ;*)
(**)
(*(*Plot nodal uncertainties*)*)
(*muncertain = SMTShowMesh["BoundaryConditions"->False,"DeformedMesh"->True,"Mesh"->Black,"FillElements"->False, "Field"->2*nodeuncertain,"ImageSize"->400,Lighting->"Neutral","Label"-> "Uncertainty predicted(2*std)",ViewPoint->{-0.44174442243945883`,1.9345565465705503`,-0.3810953769444234`},ViewVertical->{-0.015039029778453047`,0.2397723166056823`,0.9707126576762389`},"Contour"->True] ;*)
(**)
(*(*Plot gp errors*)*)
(*mgperror = SMTShowMesh["BoundaryConditions"->False,"DeformedMesh"->True,"Mesh"->Black,"FillElements"->False, "Field"->nodegperror,"ImageSize"->400,Lighting->"Neutral","Label"-> "e_gp",ViewPoint->{-0.44174442243945883`,1.9345565465705503`,-0.3810953769444234`},ViewVertical->{-0.015039029778453047`,0.2397723166056823`,0.9707126576762389`},"Contour"->True] ;*)
(**)
(*(*Plot autoencoder errors*)*)
(*mautoerror = SMTShowMesh["BoundaryConditions"->False,"DeformedMesh"->True,"Mesh"->Black,"FillElements"->False, "Field"->nodeautoerror,"ImageSize"->400,Lighting->"Neutral","Label"-> "e_reconstruction",ViewPoint->{-0.44174442243945883`,1.9345565465705503`,-0.3810953769444234`},ViewVertical->{-0.015039029778453047`,0.2397723166056823`,0.9707126576762389`},"Contour"->True] ;*)
(**)
(*(* Show plots *)*)
(*pdeformedmesh=Show[mfem,mnn,mrest,PlotRange->All,ImageSize->350]*)
(*pGPAutodisp = Show[mdispnn,"ImageSize"->400]*)
(*pfem = Show[mdispfem,"ImageSize"->400]*)
(*perrorf = Show[merror,"ImageSize"->400]*)
(*pGPAutouncertain = Show[muncertain,"ImageSize"->400]*)
(*pGPerror = Show[mgperror,"ImageSize"->400]*)
(*pAutoerror = Show[mautoerror,"ImageSize"->400]*)
(**)
(*(*Define the function to export image*)exportImage[quantity_,suffix_]:=Module[{imagePath},imagePath=StringTemplate["test_`no`/t`no`_`suffix`.jpg"][<|"no"->testnumber,"suffix"->suffix|>];*)
(*Export[imagePath,quantity,ImageResolution->300];]*)
(**)
(*(* Save all plots to the test example folder *)*)
(*exportImage[pdeformedmesh,"deformedmesh"]*)
(*exportImage[pGPAutodisp,"GpAuto_pred"]*)
(*exportImage[pfem,"fem"]*)
(*exportImage[perrorf,"GpAuto_error"]*)
(*exportImage[pGPAutouncertain,"GpAuto_uncertainty"]*)
(*exportImage[pGPerror,"Gp_error"]*)
(*exportImage[pAutoerror,"Auto_error"]*)
(**)
(*Print[StringTemplate["All plots are saved to 'test_`no`' directory"][<|"no"->testnumber|>]];*)