import dataclasses
import json
import time

import pytest

from tools.audit import AuditLog
from tools.executor import ToolCall, ToolExecutor
from tools.permissions import Principal
from tools.registry import ToolRegistry
from tools.schema import ParamSpec, Tool, ToolValidationError, validate_arguments


def make_tool(handler=None, **kwargs):
    base = dict(
        name="somar",
        description="Soma dois números.",
        handler=handler or (lambda a, b=0: a + b),
        data_access="nenhum",
        actions="nenhuma",
        parameters=(
            ParamSpec("a", "integer", "primeiro número", min_value=-100, max_value=100),
            ParamSpec("b", "integer", "segundo número", required=False, default=0),
        ),
    )
    base.update(kwargs)
    return Tool(**base)


ADMIN = Principal("ana", frozenset({"math:use", "data:write"}))
VISITOR = Principal("visitante", frozenset())


# ----------------------------------------------------------------- definição
def test_ferramenta_exige_declarar_acesso_e_acoes():
    with pytest.raises(ValueError):
        make_tool(data_access="")
    with pytest.raises(ValueError):
        make_tool(actions=" ")


def test_nome_invalido_e_parametros_repetidos():
    with pytest.raises(ValueError):
        make_tool(name="Somar!")
    with pytest.raises(ValueError):
        make_tool(parameters=(ParamSpec("a", "integer"), ParamSpec("a", "string")))
    with pytest.raises(ValueError):
        ParamSpec("x", "lista")


def test_permissao_sem_curinga():
    with pytest.raises(ValueError):
        make_tool(required_permissions=frozenset({"*"}))
    with pytest.raises(ValueError):
        Principal("x", frozenset({"admin"}))
    with pytest.raises(ValueError):
        Principal("  ")


def test_schema_publico_e_serializavel():
    schema = make_tool(required_permissions=frozenset({"math:use"})).to_schema()
    json.dumps(schema)
    assert "handler" not in schema
    assert schema["required_permissions"] == ["math:use"]
    assert schema["parameters"][0]["max"] == 100


# ---------------------------------------------------------------- validação
def test_validacao_aceita_e_aplica_padrao():
    assert validate_arguments(make_tool(), {"a": 3}) == {"a": 3, "b": 0}


@pytest.mark.parametrize("args,trecho", [
    ({}, "'a' é obrigatório"),
    ({"a": 1, "x": 2}, "desconhecidos"),
    ({"a": "1"}, "inteiro"),
    ({"a": True}, "inteiro"),          # booleano não vale como inteiro
    ({"a": 1.5}, "inteiro"),
    ({"a": 101}, "maior"),
    ({"a": -101}, "menor"),
])
def test_validacao_rejeita(args, trecho):
    with pytest.raises(ToolValidationError, match=trecho):
        validate_arguments(make_tool(), args)


def test_validacao_junta_todos_os_problemas():
    with pytest.raises(ToolValidationError) as e:
        validate_arguments(make_tool(), {"a": "x", "zzz": 1})
    assert "desconhecidos" in str(e.value) and "inteiro" in str(e.value)


def test_validacao_texto_enum_e_numeros_especiais():
    t = Tool(
        name="t", description="d", handler=lambda **k: 1, data_access="nenhum", actions="nenhuma",
        parameters=(
            ParamSpec("s", "string", max_length=5),
            ParamSpec("modo", "string", enum=("rapido", "lento"), required=False),
            ParamSpec("n", "number", required=False),
            ParamSpec("ok", "boolean", required=False),
        ),
    )
    with pytest.raises(ToolValidationError, match="limite"):
        validate_arguments(t, {"s": "123456"})
    with pytest.raises(ToolValidationError, match="valores"):
        validate_arguments(t, {"s": "a", "modo": "medio"})
    with pytest.raises(ToolValidationError, match="finito"):
        validate_arguments(t, {"s": "a", "n": float("nan")})
    with pytest.raises(ToolValidationError, match="verdadeiro"):
        validate_arguments(t, {"s": "a", "ok": "sim"})
    with pytest.raises(ToolValidationError):
        validate_arguments(t, "não é dicionário")
    assert validate_arguments(t, {"s": "ab", "modo": "lento", "n": 2, "ok": False})["ok"] is False


# ------------------------------------------------------- permissões/registro
def test_principal_e_imutavel():
    with pytest.raises(dataclasses.FrozenInstanceError):
        ADMIN.permissions = frozenset({"x:y"})


def test_registro_duplicado_e_busca():
    reg = ToolRegistry()
    reg.register(make_tool())
    assert "somar" in reg and len(reg) == 1 and reg.names() == ["somar"]
    with pytest.raises(ValueError):
        reg.register(make_tool())
    with pytest.raises(TypeError):
        reg.register("não é uma Tool")


def test_registro_filtra_por_permissao():
    reg = ToolRegistry()
    reg.register(make_tool(name="aberta"))
    reg.register(make_tool(name="restrita", required_permissions=frozenset({"data:write"})))
    assert [t.name for t in reg.available_for(VISITOR)] == ["aberta"]
    assert [t.name for t in reg.available_for(ADMIN)] == ["aberta", "restrita"]
    assert [d["name"] for d in reg.describe(VISITOR)] == ["aberta"]
    json.dumps(reg.describe())


# ------------------------------------------------------------------ executor
def setup_executor(tool=None, confirm=None, **kwargs):
    reg = ToolRegistry()
    reg.register(tool or make_tool())
    audit = AuditLog()
    return ToolExecutor(reg, audit, confirm=confirm, **kwargs), audit


def test_execucao_ok():
    ex, audit = setup_executor()
    r = ex.execute(ToolCall("somar", {"a": 2, "b": 3}), ADMIN)
    assert r.ok and r.output == 5 and r.call_id
    assert audit.entries[-1]["status"] == "ok" and audit.entries[-1]["user"] == "ana"


def test_ferramenta_inexistente():
    ex, _ = setup_executor()
    assert ex.execute(ToolCall("naoexiste", {}), ADMIN).status == "not_found"


def test_permissao_negada_nao_executa():
    chamadas = []
    tool = make_tool(handler=lambda a, b=0: chamadas.append(1),
                     required_permissions=frozenset({"data:write"}))
    ex, _ = setup_executor(tool)
    r = ex.execute(ToolCall("somar", {"a": 1}), VISITOR)
    assert r.status == "denied" and chamadas == []


def test_argumento_invalido_nao_executa():
    chamadas = []
    ex, _ = setup_executor(make_tool(handler=lambda a, b=0: chamadas.append(1)))
    r = ex.execute(ToolCall("somar", {"a": "x"}), ADMIN)
    assert r.status == "invalid" and chamadas == []


def test_modelo_nao_consegue_se_dar_permissao_via_argumentos():
    ex, _ = setup_executor(make_tool(required_permissions=frozenset({"data:write"})))
    r = ex.execute(ToolCall("somar", {"a": 1, "permissions": ["data:write"]}), VISITOR)
    assert r.status == "denied"
    r = ex.execute(ToolCall("somar", {"a": 1, "permissions": ["data:write"]}), ADMIN)
    assert r.status == "invalid"  # parâmetro desconhecido, nunca chega ao handler


def test_erro_do_handler_nao_vaza_a_mensagem():
    def falha(a, b=0):
        raise RuntimeError("senha-super-secreta-123")
    ex, audit = setup_executor(make_tool(handler=falha))
    r = ex.execute(ToolCall("somar", {"a": 1}), ADMIN)
    assert r.status == "error" and "RuntimeError" in r.error
    assert "senha-super-secreta-123" not in json.dumps(r.to_dict())
    assert "senha-super-secreta-123" not in json.dumps(audit.entries)


def test_tempo_limite():
    ex, _ = setup_executor(make_tool(handler=lambda a, b=0: time.sleep(2), timeout_seconds=0.1))
    inicio = time.perf_counter()
    r = ex.execute(ToolCall("somar", {"a": 1}), ADMIN)
    assert r.status == "timeout"
    assert time.perf_counter() - inicio < 1.0


def test_ferramenta_sensivel_sem_confirmacao_nao_executa():
    chamadas = []
    tool = make_tool(handler=lambda a, b=0: chamadas.append(1) or 1, sensitive=True)
    ex, _ = setup_executor(tool)
    r = ex.execute(ToolCall("somar", {"a": 1}), ADMIN)
    assert r.status == "needs_confirmation" and chamadas == []


def test_confirmacao_aprovada_e_recusada():
    chamadas = []
    tool = make_tool(handler=lambda a, b=0: chamadas.append(1) or 1, sensitive=True)
    ex, _ = setup_executor(tool)
    assert ex.execute(ToolCall("somar", {"a": 1}), ADMIN, confirm=lambda t, a: False).status == "rejected"
    assert chamadas == []
    visto = {}
    def aprova(t, args):
        visto.update(args)
        return True
    assert ex.execute(ToolCall("somar", {"a": 1}), ADMIN, confirm=aprova).status == "ok"
    assert visto == {"a": 1, "b": 0} and chamadas == [1]  # a confirmação vê os argumentos validados


def test_saida_grande_e_truncada():
    ex, _ = setup_executor(make_tool(handler=lambda a, b=0: "x" * 10000), max_output_chars=100)
    r = ex.execute(ToolCall("somar", {"a": 1}), ADMIN)
    assert r.ok and len(r.output) < 200 and r.output.endswith("[truncado]")


def test_saida_nao_serializavel():
    ex, _ = setup_executor(make_tool(handler=lambda a, b=0: object()))
    assert ex.execute(ToolCall("somar", {"a": 1}), ADMIN).status == "error"


# ------------------------------------------------------------------ auditoria
def test_auditoria_registra_todos_os_status_e_esconde_segredos():
    tool = Tool(
        name="login", description="d", handler=lambda user, token: "ok",
        data_access="credenciais", actions="autentica",
        parameters=(ParamSpec("user", "string"), ParamSpec("token", "string", secret=True)),
    )
    ex, audit = setup_executor(tool)
    ex.execute(ToolCall("login", {"user": "ana", "token": "TOKEN-SECRETO"}), ADMIN)
    ex.execute(ToolCall("login", {"user": "ana", "token": "TOKEN-SECRETO", "extra": "VALOR-ESTRANHO"}), ADMIN)
    ex.execute(ToolCall("inexistente", {"token": "OUTRO-SEGREDO"}), ADMIN)
    texto = json.dumps(audit.entries)
    for segredo in ("TOKEN-SECRETO", "VALOR-ESTRANHO", "OUTRO-SEGREDO"):
        assert segredo not in texto
    assert [e["status"] for e in audit.entries] == ["ok", "invalid", "not_found"]
    assert audit.entries[0]["arguments"] == {"user": "ana", "token": "***"}


def test_auditoria_em_arquivo(tmp_path):
    log = AuditLog(tmp_path / "logs" / "audit.jsonl")
    reg = ToolRegistry()
    reg.register(make_tool())
    ex = ToolExecutor(reg, log)
    ex.execute(ToolCall("somar", {"a": 1}), ADMIN)
    ex.execute(ToolCall("somar", {"a": 2}), ADMIN)
    linhas = (tmp_path / "logs" / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(linhas) == 2 and json.loads(linhas[0])["tool"] == "somar"


def test_auditoria_limita_memoria():
    log = AuditLog()
    for i in range(1500):
        log.record(i=i)
    assert len(log.entries) == 1000 and log.entries[-1]["i"] == 1499