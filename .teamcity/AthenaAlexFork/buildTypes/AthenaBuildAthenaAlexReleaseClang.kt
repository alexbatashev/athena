package AthenaAlexFork.buildTypes

import jetbrains.buildServer.configs.kotlin.v2018_2.*
import jetbrains.buildServer.configs.kotlin.v2018_2.buildFeatures.dockerSupport
import jetbrains.buildServer.configs.kotlin.v2018_2.buildSteps.exec
import jetbrains.buildServer.configs.kotlin.v2018_2.buildSteps.script

object AthenaBuildAthenaAlexReleaseClang : BuildType({
    name = "[Release][Clang] Build"

    artifactRules = "+:build_Release_Clang/athena*.tar.gz"

    params {
        param("cc", "clang-8")
        param("image_name", "llvm8")
        param("cxx", "clang++-8")
        param("repo", "athenaml/athena")
        param("env.SRC_DIR", "%system.teamcity.build.checkoutDir%")
        param("env.ATHENA_BINARY_DIR", "%teamcity.build.checkoutDir%/install_Release_Clang")
        param("env.BUILD_PATH", "%teamcity.build.checkoutDir%/build_Release_Clang")
    }

    vcs {
        root(AthenaAlexFork.vcsRoots.AthenaAlex)
    }

    steps {
        exec {
            name = "Build"
            path = "scripts/build.py"
            arguments = "--ninja --build-type Release --install-dir=%env.ATHENA_BINARY_DIR%  --use-ldd --build-type Release %env.BUILD_PATH% %env.SRC_DIR%"
            dockerImage = "registry.gitlab.com/athenaml/ubuntu_docker_images/%image_name%:latest"
            dockerPull = true
            dockerRunParameters = "-e CC=%cc% -e CXX=%cxx% -e ATHENA_BINARY_DIR=%env.ATHENA_BINARY_DIR% -e ATHENA_TEST_ENVIRONMENT=ci"
        }
        script {
            name = "Install"
            scriptContent = "cd %env.BUILD_PATH% && cmake --build . --target install;"
            dockerImage = "registry.gitlab.com/athenaml/ubuntu_docker_images/%image_name%:latest"
            dockerPull = true
        }
        script {
            name = "Pack"
            scriptContent = """cd %env.BUILD_PATH% && cpack -G "TGZ";"""
            dockerImage = "registry.gitlab.com/athenaml/ubuntu_docker_images/%image_name%:latest"
            dockerPull = true
        }
        exec {
            name = "Test"
            path = "scripts/test.sh"
            arguments = "%env.BUILD_PATH%"
            dockerImage = "registry.gitlab.com/athenaml/ubuntu_docker_images/%image_name%:latest"
            dockerPull = true
        }
    }

    features {
        dockerSupport {
            loginToRegistry = on {
                dockerRegistryId = "PROJECT_EXT_3"
            }
        }
        feature {
            type = "xml-report-plugin"
            param("xmlReportParsing.reportType", "ctest")
            param("xmlReportParsing.reportDirs", "+:build_Release_Clang/Testing/**/*.xml")
        }
    }

    requirements {
        equals("docker.server.osType", "linux")
    }
})
