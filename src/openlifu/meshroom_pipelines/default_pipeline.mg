{
    "header": {
        "pipelineVersion": "2.2",
        "releaseVersion": "2023.3.0",
        "fileVersion": "1.1",
        "template": true,
        "nodesVersions": {
            "MeshFiltering": "3.0",
            "DepthMapFilter": "4.0",
            "MeshDecimate": "1.0",
            "Meshing": "7.0",
            "FeatureExtraction": "1.3",
            "CameraInit": "9.0",
            "Texturing": "6.0",
            "ImageMatching": "2.0",
            "Publish": "1.3",
            "DepthMap": "5.0",
            "FeatureMatching": "2.0",
            "PrepareDenseScene": "3.1",
            "StructureFromMotion": "3.3"
        }
    },
    "graph": {
        "Meshing_1": {
            "nodeType": "Meshing",
            "position": [
                1600,
                0
            ],
            "inputs": {
                "input": "{DepthMapFilter_1.input}",
                "depthMapsFolder": "{DepthMapFilter_1.output}"
            }
        },
        "DepthMapFilter_1": {
            "nodeType": "DepthMapFilter",
            "position": [
                1400,
                0
            ],
            "inputs": {
                "input": "{DepthMap_1.input}",
                "depthMapsFolder": "{DepthMap_1.output}"
            }
        },
        "ImageMatching_1": {
            "nodeType": "ImageMatching",
            "position": [
                400,
                0
            ],
            "inputs": {
                "input": "{FeatureExtraction_1.input}",
                "featuresFolders": [
                    "{FeatureExtraction_1.output}"
                ]
            }
        },
        "FeatureExtraction_1": {
            "nodeType": "FeatureExtraction",
            "position": [
                200,
                0
            ],
            "inputs": {
                "input": "{CameraInit_1.output}"
            }
        },
        "StructureFromMotion_1": {
            "nodeType": "StructureFromMotion",
            "position": [
                800,
                0
            ],
            "inputs": {
                "input": "{FeatureMatching_1.input}",
                "featuresFolders": "{FeatureMatching_1.featuresFolders}",
                "matchesFolders": [
                    "{FeatureMatching_1.output}"
                ],
                "describerTypes": "{FeatureMatching_1.describerTypes}"
            }
        },
        "PrepareDenseScene_1": {
            "nodeType": "PrepareDenseScene",
            "position": [
                1000,
                0
            ],
            "inputs": {
                "input": "{StructureFromMotion_1.output}"
            }
        },
        "CameraInit_1": {
            "nodeType": "CameraInit",
            "position": [
                0,
                0
            ],
            "inputs": {}
        },
        "DepthMap_1": {
            "nodeType": "DepthMap",
            "position": [
                1200,
                0
            ],
            "inputs": {
                "input": "{PrepareDenseScene_1.input}",
                "imagesFolder": "{PrepareDenseScene_1.output}"
            }
        },
        "MeshFiltering_1": {
            "nodeType": "MeshFiltering",
            "position": [
                1800,
                0
            ],
            "inputs": {
                "inputMesh": "{Meshing_1.outputMesh}"
            }
        },
        "FeatureMatching_1": {
            "nodeType": "FeatureMatching",
            "position": [
                600,
                0
            ],
            "inputs": {
                "input": "{ImageMatching_1.input}",
                "featuresFolders": "{ImageMatching_1.featuresFolders}",
                "imagePairsList": "{ImageMatching_1.output}",
                "describerTypes": "{FeatureExtraction_1.describerTypes}"
            }
        },
        "MeshDecimate_1": {
            "nodeType": "MeshDecimate",
            "position": [
                1875,
                229
            ],
            "inputs": {
                "input": "{MeshFiltering_1.outputMesh}",
                "simplificationFactor": 0.0,
                "nbVertices": 156051
            }
        },
        "Texturing_1": {
            "nodeType": "Texturing",
            "position": [
                2178,
                24
            ],
            "inputs": {
                "input": "{Meshing_1.output}",
                "imagesFolder": "{DepthMap_1.imagesFolder}",
                "inputMesh": "{MeshDecimate_1.output}",
                "colorMapping": {
                    "enable": true,
                    "colorMappingFileType": "png"
                }
            }
        },
        "Publish_1": {
            "nodeType": "Publish",
            "position": [
                2272,
                290
            ],
            "inputs": {
                "inputFiles": [
                    "{Texturing_1.output}",
                    "{Texturing_1.outputMesh}",
                    "{Texturing_1.outputMaterial}",
                    "{Texturing_1.outputTextures}"
                ]
            }
        }
    }
}
