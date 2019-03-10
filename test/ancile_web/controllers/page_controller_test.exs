defmodule AncileWeb.PageControllerTest do
  use AncileWeb.ConnCase

  test "GET /", %{conn: conn} do
    conn = get(conn, "/")
    assert html_response(conn, 200) =~ "Welcome to Ancile!"
  end
end