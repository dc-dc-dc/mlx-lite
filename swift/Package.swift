// swift-tools-version:5.9

import PackageDescription

let package = Package(
    name: "MLXLite",
    platforms: [
        .macOS("13.3"),
    ],
    products: [
        .library(name: "MLXLite", targets: ["MLXLite"]),
        .executable(name: "MLXExample", targets: ["MLXLiteExample"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.10.0"),
    ],
    targets: [
        .target(name: "MLXLite", dependencies: [
            .product(name: "MLX", package: "mlx-swift"),
        ]),
        .executableTarget(name: "MLXLiteExample", dependencies: ["MLXLite"]),
    ]
)