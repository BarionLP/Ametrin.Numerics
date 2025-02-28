using System;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;

namespace Ametrin.Numerics.Generator;

[Generator]
[DiagnosticAnalyzer(LanguageNames.CSharp)]
public sealed class MethodVariantGenerator : DiagnosticAnalyzer, IIncrementalGenerator
{
    private static DiagnosticDescriptor NumericsExtensionsShouldEndWithTo { get; } = new(
        "AN001", "Should End with To", "Methods using GenerateVariants should end with 'To'", "Naming", DiagnosticSeverity.Warning, isEnabledByDefault: true
    );

    public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } = [NumericsExtensionsShouldEndWithTo];

    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        var nodes = context.SyntaxProvider.CreateSyntaxProvider(
            (node, c) => node is ClassDeclarationSyntax { AttributeLists.Count: > 0 },
            (context, token) => context.SemanticModel.GetDeclaredSymbol(context.Node, token) as INamedTypeSymbol
        );

        context.RegisterSourceOutput(nodes.Combine(context.CompilationProvider), (context, p) =>
        {
            var type = p.Left;
            var compilation = p.Right;
            if (type is null) return;

            var attribute = type!.GetAttributes().FirstOrDefault(a => a.AttributeClass is { Name: "NumericsHelperAttribute", ContainingAssembly.Name: "Ametrin.Numerics", TypeArguments.Length: 1 });
            if (attribute is null) return;

            var methods = type.GetMembers().OfType<IMethodSymbol>().Where(s => s.Parameters.Length > 1 && s.GetAttributes().Any(a => a.AttributeClass is { Name: "GenerateVariantsAttribute", ContainingAssembly.Name: "Ametrin.Numerics" })).ToImmutableArray();
            if (methods.IsEmpty) return;

            var tensorType = attribute.AttributeClass!.TypeArguments[0];

            var importFromTensorPrimitives = attribute.NamedArguments.First(p => p.Key is "GenerateFromTensorPrimitives").Value.Values.Select(v => v.Value).OfType<string>() ?? [];

            var sb = new StringBuilder();

            sb.Append($$"""
            namespace {{type.ContainingNamespace}};

            static partial class {{type.Name}}
            {
            """);

            var tp = compilation.GetTypeByMetadataName("System.Numerics.Tensors.TensorPrimitives")!;

            foreach (var methodName in importFromTensorPrimitives)
            {
                var method = tp.GetMembers(methodName).OfType<IMethodSymbol>().Where(s => s is { IsStatic: true, DeclaredAccessibility: Accessibility.Public, IsGenericMethod: false }).First();
                var parameters = method.Parameters.Take(method.Parameters.Length - 1);
                var parametersStrings = method.Parameters.Select(p => $"{(p.Type is { Name: "Span" or "ReadOnlySpan", ContainingNamespace.Name: "System" } ? tensorType : p.Type)} {p.Name}");

                sb.AppendLine($$"""

                #region {{method.Name}}
                public static void {{method.Name}}To({{string.Join(", ", parametersStrings)}})
                {
                    System.Numerics.Tensors.TensorPrimitives.{{methodName}}({{string.Join(", ", method.Parameters.Select(p => p.Type is { Name: "Span" or "ReadOnlySpan", ContainingNamespace.Name: "System" } ? $"{p.Name}.AsSpan()" : p.Name))}});
                }
                public static void {{method.Name}}ToSelf({{(method.IsExtensionMethod ? "this " : "")}}{{string.Join(", ", parametersStrings.Take(method.Parameters.Length - 1))}})
                {
                    {{method.Name}}To({{string.Join(", ", parameters.Select(p => p.Name))}}, {{method.Parameters[0].Name}});
                }
                public static {{tensorType}} {{method.Name}}({{(method.IsExtensionMethod ? "this " : "")}}{{string.Join(", ", parametersStrings.Take(method.Parameters.Length - 1))}})
                {
                    var destination = {{tensorType}}.OfSize({{method.Parameters[0].Name}});
                    {{method.Name}}To({{string.Join(", ", parameters.Select(p => p.Name))}}, destination);
                    return destination;
                }
                #endregion
            """);
            }

            foreach (var method in methods)
            {
                var accessibility = method.DeclaredAccessibility;
                var parameters = method.Parameters.Take(method.Parameters.Length - 1);
                sb.AppendLine($$"""
                
                #region {{method.Name}}
                {{accessibility.ToString().ToLowerInvariant()}} {{(method.IsStatic ? "static" : "")}} void {{method.Name}}Self({{(method.IsExtensionMethod ? "this " : "")}}{{string.Join(", ", parameters)}})
                {
                    {{method.Name}}({{string.Join(", ", parameters.Select(p => p.Name))}}, {{method.Parameters[0].Name}});
                }
                {{accessibility.ToString().ToLowerInvariant()}} {{(method.IsStatic ? "static" : "")}} {{method.Parameters[0].Type}} {{method.Name.Substring(0, method.Name.Length - 2)}}({{(method.IsExtensionMethod ? "this " : "")}}{{string.Join(", ", parameters)}})
                {
                    var destination = {{method.Parameters[0].Type}}.OfSize({{method.Parameters[0].Name}});
                    {{method.Name}}({{string.Join(", ", parameters.Select(p => p.Name))}}, destination);
                    return destination;
                }
                #endregion
            """);
            }

            sb.AppendLine("}");

            context.AddSource($"{type.Name}.g.cs", sb.ToString());
        });
    }

    public override void Initialize(AnalysisContext context)
    {
        context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.Analyze | GeneratedCodeAnalysisFlags.ReportDiagnostics);
        context.EnableConcurrentExecution();
        context.RegisterSymbolAction(context =>
        {
            if (context.Symbol is not IMethodSymbol method) return;

            if (!method.Name.EndsWith("To") && method.GetAttributes().Any(a => a.AttributeClass is { Name: "GenerateVariantsAttribute", ContainingAssembly.Name: "Ametrin.Numerics" }))
            {
                context.ReportDiagnostic(Diagnostic.Create(NumericsExtensionsShouldEndWithTo, context.Symbol.Locations[0]));
            }

        }, SymbolKind.Method);
    }
}
